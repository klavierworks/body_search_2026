# INPUT

Datasets go here, one folder per dataset. Both halves of the project resolve
paths relative to this directory, so a dataset is referred to by name:

    bodypose run output-images        # reads INPUT/output-images

The images do not have to physically live here. A symlink is the normal case,
and keeps the repository small while letting the images sit on whatever volume
suits:

    ln -s /Volumes/SS2_OSX/output-images   INPUT/output-images
    ln -s /Volumes/SS2_OSX/DATASETS_BACKUP INPUT/datasets-backup

Because a dataset is identified by name rather than by absolute path, an index
built against one location still resolves after you move the images — repoint
the symlink at a local copy and nothing needs rebuilding.

`bodypose datasets` lists what is here and what has been indexed.
