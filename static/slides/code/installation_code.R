install.packages("devtools")
library(devtools)
install.packages("reticulate")
library(reticulate)
install.packages("tensorflow")
library(tensorflow)
install_github("rstudio/keras3") 
library(keras3)
install_keras(gpu = FALSE)

# Check TensorFlow version
cat("TensorFlow version: ", tf$`__version__`, "\n")

# Test if TensorFlow is built with CUDA
cat("TensorFlow is built with CUDA: ", tf$config$experimental$is_built_with_cuda(), "\n")

# List all available devices
cat("All devices: \n")
print(tf$config$list_physical_devices(device_type = NULL))

# List GPU devices
cat("GPU devices: \n")
print(tf$config$list_physical_devices(device_type = 'GPU'))

# Print a randomly generated tensor and its reduced sum
# Generates a random tensor and calculates the sum of elements
random_tensor <- tf$random$normal(shape = as.integer(c(1, 10)))
random_tensor
cat("Reduced sum of random tensor: ", tf$math$reduce_sum(random_tensor)$numpy(), "\n")


