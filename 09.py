import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def generate_synthetic_data(n_samples, noise):
    np.random.seed(0)
    X = np.random.rand(n_samples) * 10  # Random values between 0 and 10
    Y = 3 * X + 7 + np.random.randn(n_samples) * noise  # Linear relation with noise
    return X, Y

def locally_weighted_regression(x0, X, Y, tau):
    m = X.shape[0]
    x0 = np.r_[1, x0]  # Add intercept term
    X = np.c_[np.ones(m), X]  # Add intercept term
    
    # Calculate weights
    W = np.exp(-np.sum((X - x0)**2, axis=1) / (2 * tau**2))
    
    # Compute the theta values using normal equation
    theta = np.linalg.inv(X.T @ (W[:, None] * X)) @ (X.T @ (W * Y))
    
    # Predict the value at x0
    y0 = x0 @ theta
    return y0

def main():
    st.title("BYTES BRIGADE")
    st.title("Locally Weighted Regression")
    
    # Generate synthetic dataset
    X, Y = generate_synthetic_data(n_samples=100, noise=10.0)
    
    st.subheader("Generated Dataset")
    data = np.vstack((X, Y)).T
    st.write(pd.DataFrame(data, columns=["Feature", "Target"]))
    
    # Define the range of x values for prediction
    x_pred = np.linspace(X.min(), X.max(), 300)
    
    # User input for bandwidth parameter tau
    tau = st.slider("Select Bandwidth (tau)", 0.01, 1.0, 0.1)
    
    # Predict y values using locally weighted regression
    y_pred = np.array([locally_weighted_regression(x, X, Y, tau) for x in x_pred])
    
    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, Y, color='blue', label='Data Points')
    ax.plot(x_pred, y_pred, color='red', label='LWR Curve')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.legend()
    ax.set_title('Locally Weighted Regression')
    
    # Display plot in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    main()
