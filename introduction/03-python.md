# Python for ML Quick Reference

## Essential ML Packages

### The Big Three
```python
import numpy as np          # Numbers & arrays
import pandas as pd         # Data tables  
import matplotlib.pyplot as plt  # Plotting
```

## NumPy Basics
```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros(10)
random_data = np.random.randn(100)  # 100 random numbers

# Basic stats
print(f"Mean: {arr.mean()}")
print(f"Std: {arr.std()}")
print(f"Max: {arr.max()}")
```

## Pandas Basics
```python
import pandas as pd

# Create a simple dataframe
data = {
    'age': [25, 30, 35, 40],
    'salary': [50000, 60000, 75000, 90000]
}
df = pd.DataFrame(data)

# Quick exploration
df.head()        # First 5 rows
df.describe()    # Statistics summary
df.info()        # Data types and memory
```

## Matplotlib Basics
```python
import matplotlib.pyplot as plt
import numpy as np

# Simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("My First Plot")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.title("Distribution")
plt.show()
```

## Loading Data
```python
# CSV file
df = pd.read_csv('data.csv')

# From URL
url = 'https://example.com/data.csv'
df = pd.read_csv(url)

# Check what you loaded
print(df.shape)  # (rows, columns)
print(df.columns)  # Column names
```

## Package Management
```python
# Install packages (in notebook)
!pip install scikit-learn

# Check installed packages
!pip list

# Check specific package version
import numpy
print(numpy.__version__)
```

## Common Patterns for ML
```python
# 1. Import everything at the top
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 2. Set random seed for reproducibility
np.random.seed(42)

# 3. Check your data shape constantly
print(f"Data shape: {data.shape}")

# 4. Visualize before modeling
plt.figure(figsize=(10, 6))  # Bigger plots
```

## Debugging Tips
```python
# Print variable type
print(type(variable))

# Print shape (for arrays/dataframes)
print(variable.shape)

# Check for missing values
print(df.isnull().sum())

# Quick variable check
print(f"Variable x: {x}")
```

## Memory Management
```python
# Delete large variables when done
del large_dataset

# Check memory usage of dataframe
df.memory_usage(deep=True)
```

## Common Errors & Fixes
- **NameError**: Variable not defined → Run cells in order
- **ModuleNotFoundError**: Package missing → `!pip install package_name`
- **AttributeError**: Wrong method name → Check spelling/documentation
- **ValueError**: Wrong data type/shape → Print shape and type to debug