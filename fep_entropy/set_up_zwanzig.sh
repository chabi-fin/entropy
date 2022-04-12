#!/bin/bash

systems=("000" "010" "020" "030" "040" "050" "060" "070" "080" "090" "100")

# Rerun the simulation in new topology
# Forward direction
for i in $(seq 0 9); do
  gmx18 mdrun -s lambda_${systems[$((i+1))]}/md_100ns.tpr \
    -rerun lambda_${systems[$i]}/md_100ns.xtc \
    -e forward/${systems[$((i+1))]}_in_${systems[$i]}.edr -nobackup
done

# Reverse direction
for i in $(seq 1 10); do
  gmx18 mdrun -s lambda_${systems[$((i-1))]}/md_100ns.tpr \
    -rerun lambda_${systems[$i]}/md_100ns.xtc \
    -e reverse/${systems[$((i-1))]}_in_${systems[$i]}.edr -nobackup
done

##########################################################

# Extract the internal energies
# Forward
for i in $(seq 0 9); do
  # Shift frame
  echo "Potential" | gmx18 energy \
    -f forward/${systems[$((i+1))]}_in_${systems[$i]}.edr \
    -o forward/${systems[$((i+1))]}_in_${systems[$i]}.xvg -nobackup
  # Same frame
  echo "Potential" | gmx18 energy \
    -f lambda_${systems[$i]}/md_100ns.edr \
    -o forward/${systems[$i]}_in_${systems[$i]}.xvg -nobackup
done

# Reverse
for i in $(seq 1 10); do
  # Shift framecd
  echo "Potential" | gmx18 energy \
    -f reverse/${systems[$((i-1))]}_in_${systems[$i]}.edr \
    -o reverse/${systems[$((i-1))]}_in_${systems[$i]}.xvg -nobackup
  # Same frame
  echo "Potential" | gmx18 energy \
    -f lambda_${systems[$i]}/md_100ns.edr \
    -o reverse/${systems[$i]}_in_${systems[$i]}.xvg -nobackup
	echo ${systems[$i]}
done
