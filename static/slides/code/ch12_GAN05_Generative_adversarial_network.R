library(keras)
library(tensorflow)
library(tfdatasets)
library(tfautograph)
# Load, resize, and format pictures into appropriate arrays
preprocess_image <- tf_function(function(image_path) {
  image_path %>%
    tf$io$read_file() %>%
    tf$io$decode_image() %>%
    # Add batch axis, equivalent to .[tf$newaxis, all_dims()]. axis arg is 0-based.
    tf$expand_dims(axis = 0L) %>%
    # Cast from 'uint8'.
    tf$cast("float32") %>%
    inception_v3_preprocess_input()
})

# Convert a tensor array into a valid image and undo preprocessing 
deprocess_image <- tf_function(function(img) {
  img %>%
    # Drop first dim—the batch axis (must be size 1), the inverse of tf$expand_dims().
    tf$squeeze(axis = 0L) %>%
    # Rescale so values in [-1, 1] are remapped to [0, 255].
    { (. * 127.5) + 127.5 } %>%
    # saturate_case() clips values to the dtype range: [0, 255].
    tf$saturate_cast("uint8")
})


display_image_tensor <- function(x, ..., max = 255,
                                 plot_margins = c(0, 0, 0, 0)) {

  if (!is.null(plot_margins))
    # Default to no margins when plotting images.
    withr::local_par(mar = plot_margins)

  x %>%
    # Convert tensors to R arrays.
    as.array() %>%
    drop() %>%
    # Convert to R native raster format.
    as.raster(max = max) %>%
    plot(..., interpolate = FALSE)
}

# # Install gdown.
# reticulate::py_install("gdown", pip = TRUE)
# # Download the compressed data using gdown.
# system("gdown 1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684")
# # Uncompress the data.
# zip::unzip("img_align_celeba.zip", exdir = "celeba_gan")
# 

dataset <- image_dataset_from_directory(
  "celeba_gan",
  # Only the images will be returned—no labels.
  label_mode = NULL,
  # We will resize the images to 64 × 64 by using a smart combination of cropping 
  # and resizing to preserve aspect ratio. We don’t want face proportions to get distorted!
  image_size = c(64, 64),
  batch_size = 32,
  crop_to_aspect_ratio = TRUE
)

# rescale the images to the [0-1] range
dataset %<>% dataset_map(~ .x / 255)

# Displaying the first image
# x <- dataset %>% as_iterator() %>% iter_next()
# display_image_tensor(x[1, , , ], max = 1)


discriminator <-
  keras_model_sequential(name = "discriminator",
                         input_shape = c(64, 64, 3)) %>%
  layer_conv_2d(64, kernel_size = 4, strides = 2, padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d(128, kernel_size = 4, strides = 2, padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d(128, kernel_size = 4, strides = 2, padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_flatten() %>%
  # One dropout layer: an important trick!
  layer_dropout(0.2) %>%
  layer_dense(1, activation = "sigmoid")

discriminator

# The latent space will be made of 128-dimensional vectors.
latent_dim <- 128

generator <-
  keras_model_sequential(name = "generator",
                         input_shape = c(latent_dim)) %>%
  # Produce the same number of coefficients we had at the level of
  # the Flatten layer in the encoder.
  layer_dense(8 * 8 * 128) %>%
  # Revert the layer_flatten() of the encoder.
  layer_reshape(c(8, 8, 128)) %>%
  # Revert the layer_conv_2d() of the encoder.
  layer_conv_2d_transpose(128, kernel_size = 4, strides = 2,
                          padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d_transpose(256, kernel_size = 4, strides = 2,
                          padding = "same") %>%
  # Use Leaky Relu as our activation
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d_transpose(512, kernel_size = 4, strides = 2,
                          padding = "same") %>%
  layer_activation_leaky_relu(alpha = 0.2) %>%
  layer_conv_2d(3, kernel_size = 5, padding = "same",
                activation = "sigmoid")

generator

model_gan <- new_model_class(
  classname = "GAN",

  initialize = function(discriminator, generator, latent_dim) {
    super$initialize()
    self$discriminator  <- discriminator
    self$generator      <- generator
    self$latent_dim     <- as.integer(latent_dim)
    # Set up metrics to track the two losses over each training epoch
    self$d_loss_metric  <- metric_mean(name = "d_loss")
    self$g_loss_metric  <- metric_mean(name = "g_loss")
  },

  compile = function(d_optimizer, g_optimizer, loss_fn) {
    super$compile()
    self$d_optimizer <- d_optimizer
    self$g_optimizer <- g_optimizer
    self$loss_fn <- loss_fn
  },

   metrics = mark_active(function() {
      list(self$d_loss_metric,
           self$g_loss_metric)
   }),

  # train_step is called with a batch of real images.
  train_step = function(real_images) {
    batch_size <- tf$shape(real_images)[1]
    # Sample random points in the latent space.
    random_latent_vectors <-
      tf$random$normal(shape = c(batch_size, self$latent_dim))
    # Decode them to fake images.
    generated_images <- self$generator(random_latent_vectors)

    # Combine them with real images.
    combined_images <-
      tf$concat(list(generated_images,
                     real_images),
                axis = 0L)

    # Assemble labels, discriminating real from fake images.
    labels <-
      tf$concat(list(tf$ones(tuple(batch_size, 1L)),
                     tf$zeros(tuple(batch_size, 1L))),
                axis = 0L)

    # Add random noise to the labels—an important trick!
    labels %<>% `+`(
      tf$random$uniform(tf$shape(.), maxval = 0.05))

    with(tf$GradientTape() %as% tape, {
      predictions <- self$discriminator(combined_images)
      d_loss <- self$loss_fn(labels, predictions)
    })

    grads <- tape$gradient(d_loss, self$discriminator$trainable_weights)
    # Train the discriminator.
    self$d_optimizer$apply_gradients(
      zip_lists(grads, self$discriminator$trainable_weights))

    # Sample random points in the latent space.
    random_latent_vectors <-
      tf$random$normal(shape = c(batch_size, self$latent_dim))

    # Assemble labels that say “these are all real images” (it’s a lie!).
    misleading_labels <- tf$zeros(tuple(batch_size, 1L))

    with(tf$GradientTape() %as% tape, {
      predictions <- random_latent_vectors %>%
        self$generator() %>%
        self$discriminator()
      g_loss <- self$loss_fn(misleading_labels, predictions)
    })
    grads <- tape$gradient(g_loss, self$generator$trainable_weights)
    # Train the generator.
    self$g_optimizer$apply_gradients(
      zip_lists(grads, self$generator$trainable_weights))

    self$d_loss_metric$update_state(d_loss)
    self$g_loss_metric$update_state(g_loss)

    list(d_loss = self$d_loss_metric$result(),
         g_loss = self$g_loss_metric$result())
  })

callback_gan_monitor <- new_callback_class(
  classname = "GANMonitor",

  initialize = function(num_img = 3, latent_dim = 128,
                        dirpath = "gan_generated_images") {
    private$num_img <- as.integer(num_img)
    private$latent_dim <- as.integer(latent_dim)
    private$dirpath <- fs::path(dirpath)
    fs::dir_create(dirpath)
  },

  on_epoch_end = function(epoch, logs = NULL) {
    random_latent_vectors <-
      tf$random$normal(shape = c(private$num_img, private$latent_dim))

    generated_images <- random_latent_vectors %>%
      self$model$generator() %>%
      # Scale and clip to uint8 range of [0, 255], and cast to uint8.
      { tf$saturate_cast(. * 255, "uint8") }

    for (i in seq(private$num_img))
      tf$io$write_file(
        filename = private$dirpath / sprintf("img_%03i_%02i.png", epoch, i),
        contents = tf$io$encode_png(generated_images[i, , , ])
      )
  }
)

epochs <- 100

gan <- model_gan(discriminator = discriminator,
                 generator = generator,
                 latent_dim = latent_dim)

gan %>% compile(
  d_optimizer = optimizer_adam(learning_rate = 0.0001),
  g_optimizer = optimizer_adam(learning_rate = 0.0001),
  loss_fn = loss_binary_crossentropy()
)


gan %>% fit(
  dataset,
  epochs = epochs,
  callbacks = callback_gan_monitor(num_img = 10, latent_dim = latent_dim)
)

