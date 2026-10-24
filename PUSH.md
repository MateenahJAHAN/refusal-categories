# How to get this on GitHub

The repo is already a git repository with one commit made. You just need to point it at
GitHub and push.

## Option A — command line (2 minutes)

1. Create an **empty** repo on GitHub named `refusal-categories` (no README, no .gitignore,
   no licence — this bundle already has them).
2. Unzip this bundle, then from inside the folder:

```bash
git remote set-url origin https://github.com/MateenahJAHAN/refusal-categories.git
git branch -M main
git push -u origin main
```

If git asks for a password, use a **personal access token**, not your account password:
GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens →
Generate new token → give it `Contents: Read and write` on this one repo.

## Option B — no command line

1. Create the empty repo on GitHub as above.
2. On the new repo's page click **uploading an existing file**.
3. Drag in `README.md`, `LICENSE`, `requirements.txt`, and the `notebooks/` and `docs/`
   folders. Commit.

This loses the commit history but the content is identical.

## After pushing

- Pin the repo on your GitHub profile (BlueDot explicitly recommends this).
- The repo URL is what Unit 5's submission form wants in the "Link to your project" field —
  but only once Units 3 and 4 are actually done.
