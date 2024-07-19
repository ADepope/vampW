import matplotlib.pyplot as plt

def plot_metrics(corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, dl_dmus, a, p, correct_mu, correct_alpha, n, m):
    plt.figure(figsize=(20, 10))
    
    # Overall title
    plt.suptitle(f"Working with {n}x{m} matrix", fontsize=16)

    # Plotting corr_x
    plt.subplot(3, 4, 1)
    plt.ylabel('corr_x')
    plt.plot(range(len(corrs_x)), corrs_x, 'ro-')

    # Plotting l2_err_x
    plt.subplot(3, 4, 2)
    plt.ylabel('l2_err_x')
    plt.plot(range(len(l2_errs_x)), l2_errs_x, 'ro-')

    # Plotting corr_z
    plt.subplot(3, 4, 3)
    plt.ylabel('corr_z')
    plt.plot(range(len(corrs_z)), corrs_z, 'bo-')

    # Plotting l2_err_z
    plt.subplot(3, 4, 4)
    plt.ylabel('l2_err_z')
    plt.plot(range(len(l2_errs_z)), l2_errs_z, 'bo-')

    # Plotting mu evolution
    plt.subplot(3, 4, 5)
    plt.ylabel('mu')
    plt.plot(range(len(mus)), mus, 'go-')
    plt.axhline(y=correct_mu, color='r', linestyle='--', label='Correct mu')
    plt.legend()

    # Plotting alpha evolution
    plt.subplot(3, 4, 6)
    plt.ylabel('alpha')
    plt.plot(range(len(alphas)), alphas, 'go-')
    plt.axhline(y=correct_alpha, color='r', linestyle='--', label='Correct alpha')
    plt.legend()

    # Plotting Actual vs Predicted Scatter Plot in the last cell
    plt.subplot(3, 4, 7)
    indices = range(len(a))
    plt.scatter(indices, a, color='blue', label='Actual')
    plt.scatter(indices, p, color='red', label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted Scatter Plot')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make space for the suptitle
    plt.show()
