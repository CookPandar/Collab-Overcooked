INSTALLER="/mnt/volumes/ss-sai-bd-ga/zhangshuwen/Miniconda3-latest-Linux-x86_64.sh"
PREFIX="/mnt/volumes/ss-sai-bd-ga/zhangshuwen/miniconda3"
export PS1="${PS1:-\u@\h:\w\$ }"
: "${debian_chroot:=}"
PS1="${debian_chroot:+($debian_chroot)}$PS1"

if [ ! -f "$INSTALLER" ]; then
  echo "Installer not found: $INSTALLER"
  exit 1
fi

bash "$INSTALLER" -b -u -p "$PREFIX"

export PATH="$PREFIX/bin:$PATH"
if [ -f "$PREFIX/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "$PREFIX/etc/profile.d/conda.sh"
fi
if ! grep -q "export PATH=\"$PREFIX/bin:\$PATH\"" "$HOME/.bashrc" 2>/dev/null; then
  echo "export PATH=\"$PREFIX/bin:\$PATH\"" >> "$HOME/.bashrc"
fi

echo "Miniconda installed at $PREFIX and initialized."
curl -s -L https://lizrcache.ssai.lixiangoa.com/lizr-cluster-v2/down_lizrun.sh | bash
curl -s https://gitlabee.chehejia.cQom/lapi/public/lpai-asset-py-sdk/-/raw/master/scripts/adjust_pip.sh | sh -s 1
curl -s https://gitlabee.chehejia.com/lpai/public/lpai-asset-py-sdk/-/raw/master/scripts/adjust_conda.sh | sh
curl -s https://gitlabee.chehejia.com/lpai/public/lpai-asset-py-sdk/-/raw/master/scripts/adjust_apt.sh | sh

HELPER="$PREFIX/bin/run_conda_env.sh"
mkdir -p "$(dirname "$HELPER")"
cat >"$HELPER" <<EOF
#!/bin/bash
set -euo pipefail
if [ \$# -lt 1 ]; then
  echo "Usage: \$0 <env_prefix> [command ...]" >&2
  exit 1
fi
ENV_PREFIX=\$1
shift
source "$PREFIX/etc/profile.d/conda.sh"
conda activate "\$ENV_PREFIX"
if [ \$# -eq 0 ]; then
  exec bash --login
else
  exec "\$@"
fi
EOF
chmod +x "$HELPER"
echo "Helper script created at $HELPER. Use it to run commands inside any conda env, e.g.:"
echo "  $HELPER /path/to/.cluster_env python ..."
