"""Wall object for environment simulation."""


class Wall:
    """Represents a wall in the simulation environment."""

    def __init__(self, x, y, z, length, height, thickness, orientation=0):
        """
        Initialize a wall object.

        Args:
            x: X coordinate of the wall center
            y: Y coordinate of the wall center
            z: Z coordinate of the wall center
            length: Length of the wall
            height: Height of the wall
            thickness: Thickness of the wall
            orientation: Rotation angle in radians (default: 0)
        """
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.height = height
        self.thickness = thickness
        self.orientation = orientation

    def __repr__(self):
        """Return string representation of the wall."""
        return (
            f"Wall(x={self.x}, y={self.y}, z={self.z}, "
            f"length={self.length}, height={self.height}, "
            f"thickness={self.thickness}, orientation={self.orientation})"
        )

    def get_position(self):
        """Return the wall's position as a tuple."""
        return (self.x, self.y, self.z)

    def get_dimensions(self):
        """Return the wall's dimensions as a tuple."""
        return (self.length, self.height, self.thickness)

    def set_position(self, x, y, z):
        """Update the wall's position."""
        self.x = x
        self.y = y
        self.z = z

    def set_orientation(self, orientation):
        """Update the wall's orientation in radians."""
        self.orientation = orientation
