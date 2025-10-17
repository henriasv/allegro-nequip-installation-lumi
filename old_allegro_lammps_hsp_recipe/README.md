# allegro-nequip-installation-lumi

Change directory to `allegro_lammps_hsp_recipe`

In `container_builder.conf`, change the date of the container name suffix to today (or any other date you want in the name of your container). This is to avoid overwriting existing versions of the container.

Run 
```
chmod +x container_builder.sh
./container_builder.sh
```

If all goes well, to install the container as a module, run the following commands:

```
module purge
module load LUMI/24.03 partition/container EasyBuild-user
eb -r. -f $easyconfig
```