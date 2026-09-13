NAME          RNGINT
ROWS
 N  COST
 L  R_L
 G  R_G
 E  R_E
COLUMNS
    X_CONT    COST      1.00000   R_L       1.00000
    X_CONT    R_G       1.00000
    MARK01    'MARKER'            'INTORG'
    X_INT     COST      2.00000   R_E       1.00000
    X_INT     R_L       2.00000
    MARK02    'MARKER'            'INTEND'
    X_BIN     COST      3.00000   R_G       2.00000
    X_BIN     R_E       1.00000
RHS
    RHS1      R_L       10.0000   R_G       5.00000
    RHS1      R_E       8.00000
RANGES
    RNG1      R_L       3.00000   R_G       4.00000
    RNG1      R_E       2.00000
BOUNDS
 UI BND1      X_INT     10.0000
 BV BND1      X_BIN
ENDATA
