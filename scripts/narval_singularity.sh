module load apptainer

DATA_DIR=/home/mozaffar/projects/def-mmehride/mozaffar/data

rm -rf $SLURM_TMPDIR/torch-one-shot.sif;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif;
tar -xf /home/mozaffar/projects/def-mmehride/mozaffar/torch-one-shot.tar -C $SLURM_TMPDIR;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs;
cp /etc/ssl/certs/ca-bundle.crt ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs/ca-bundle.crt;

singularity exec \
    --bind $PWD:/home/mozaffar \
    --bind $SLURM_TMPDIR:/tmp \
    --nv \
    ${SLURM_TMPDIR}/torch-one-shot.sif \
    mkdir -p /home/mozaffar/data

singularity shell \
    --bind $PWD:/home/mozaffar \
    --bind $SLURM_TMPDIR:/tmp \
    --bind $DATA_DIR:/home/mozaffar/data \
    --nv ${SLURM_TMPDIR}/torch-one-shot.sif 