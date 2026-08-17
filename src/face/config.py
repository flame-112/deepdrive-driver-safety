"""Configuration shared by the face-landmark milestone."""

# MediaPipe Face Mesh landmark identifiers for the visible eye contours and lips.
# These regions are visualized now and will later support EAR/MAR calculations.
LEFT_EYE_INDICES: tuple[int, ...] = (33, 133, 160, 158, 153, 144)
RIGHT_EYE_INDICES: tuple[int, ...] = (362, 263, 387, 385, 380, 373)
MOUTH_INDICES: tuple[int, ...] = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
)

FACE_MESH_LANDMARK_COUNT = 468
