## Applying the patch

The optimization patch is provided at `patches/opt.patch`.

It is intended to be applied on top of the parent commit of `5a1dcb5`.

Commands:

git checkout 5a1dcb5^
git apply --check patches/opt.patch
git apply patches/opt.patch
