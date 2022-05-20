# Image Reconstruction Repo
Please pull and install the repo as editable library at the directory root:
```bash
git remote update
git pull
pip install -e .
```
And copy `2d_knee.mat` and `2dt_heart.mat` into `data/`
I have used virtualenv for IML projects because I think we use similar packages thereof:
```bash
pip install -r requirements.txt
```
Please run `python scripts/test_repo_initialization.py` to see whether everything is properly initialized.
