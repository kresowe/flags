from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from image_preprocessor import get_pixels_positions

if __name__ == '__main__':
    X_MAX = 32
    Y_MAX = 20

    pixels_positions, width_limits, height_limits = get_pixels_positions(width=X_MAX, height=Y_MAX)

    current_axis = plt.gca()
    for pixel_position in pixels_positions:
        current_axis.add_patch(Rectangle((pixel_position[1], pixel_position[0]), 1, 1))


    plt.axis('scaled')
    plt.axis([0, X_MAX, 0, Y_MAX])
    for y_lim in height_limits:
        plt.axhline(y_lim)
    for x_lim in width_limits:
        plt.axvline(x_lim)
    plt.savefig('img/pixels.pdf', bbox_inches='tight')
    plt.savefig('img/pixels.png', bbox_inches='tight', dpi=80)
    plt.show()
