# MuJoCo Ant Hierarchical

This directory tests a hierarchical Ant route that freezes the pretrained `jren123/sac-ant-v4` locomotion policy and only changes the high-level navigation interface.

The first attempted method is target-aligned observation transformation:

1. Compute the heading from current Ant xy position to the current goal/waypoint.
2. Rotate the root quaternion and root xy velocity in the Ant-v4 observation into that target-aligned frame.
3. Feed the transformed observation into the frozen forward locomotion policy.
4. Apply the frozen policy action unchanged to the real MuJoCo Ant.

No joint-level locomotion retraining is done.
