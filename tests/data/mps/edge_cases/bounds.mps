NAME          BOUNDSTST
ROWS
 N  COST
 L  ROW1
COLUMNS
    X_DEF     COST      1.00000   ROW1      1.00000
    X_LO      COST      2.00000   ROW1      1.00000
    X_UP      COST      3.00000   ROW1      1.00000
    X_FX      COST      4.00000   ROW1      1.00000
    X_FR      COST      5.00000   ROW1      1.00000
    X_MI      COST      6.00000   ROW1      1.00000
    X_PL      COST      7.00000   ROW1      1.00000
RHS
    RHS1      ROW1      100.000
BOUNDS
 LO BND1      X_LO      2.50000
 UP BND1      X_UP      5.00000
 FX BND1      X_FX      7.00000
 FR BND1      X_FR
 MI BND1      X_MI
 UP BND1      X_MI      3.00000
 PL BND1      X_PL
ENDATA
