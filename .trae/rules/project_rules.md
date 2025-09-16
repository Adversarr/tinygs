# C/C++/CUDA

## Comment Style
Add concise but useful function-level comments when generating code in header files.

**Simple Functions**(e.g. getter/setter, get_params/set_params, ...): prefer very short comment/doc string.

```cpp
/// @brief Describe the functionality
ret_t simple_function();
```

**Complex Functions Declaration**(e.g. core algorithms, complex logic, ...): provide comments/doc string. Provide the function signature and parameter descriptions, if the functions' signature really has this complexity:
```cpp
/**
 * @brief Describe the functionality
 *
 * @param param1 Description of param1
 * @param param2 Description of param2
 * @return Return value description
 */
ret_t complex_function(...);
```

However, if the function's parameters' naming is self-explanatory, no need to add comments:
```cpp
/// @brief Save the image to disk
void save_image(const uint8_t* data, int width, int height, int channels, const std::string& filename);
```

**Functions Implementation**(e.g. core algorithms, complex logic, ...): provide comments/doc string.

```
ret_t complex_function_implementation(...) {
  // Step 1: Description of step 1
  step_1(...);
  ...
  // Step 2: Description of step 2
  step_2(...);
  ...
  // ...
}
```

**Separator**: For large functions/classes, add a separator between different sections.

```cpp
////////////////////////////// Getter/Setter //////////////////////////////
...

////////////////////////////// Algorithms //////////////////////////////
...
```

