"""Configuration shared by the face-landmark milestone."""

# MediaPipe Face Mesh uses the subject's left/right (the person's own left eye).
# These six points per eye are the visible contour used for drawing and EAR.
# Order for EAR tuples is p1..p6 from Soukupová & Čech (2016):
# p1, p4 = eye corners (horizontal);
# p2, p3 = upper eyelid; p6, p5 = matching lower eyelid.
LEFT_EYE_INDICES: tuple[int, ...] = (362, 263, 387, 385, 380, 373)
RIGHT_EYE_INDICES: tuple[int, ...] = (33, 133, 160, 158, 153, 144)
LEFT_EYE_EAR_INDICES: tuple[int, int, int, int, int, int] = (362, 387, 385, 263, 380, 373)
RIGHT_EYE_EAR_INDICES: tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)

# Mouth Aspect Ratio (same p1..p6 geometry as EAR):
# p1, p4 = mouth corners; p2, p6 = inner lips; p3, p5 = outer lip top/bottom.
MOUTH_MAR_INDICES: tuple[int, int, int, int, int, int] = (61, 13, 0, 291, 17, 14)
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
