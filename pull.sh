#!/bin/sh
usage() {
	cat <<EOF
usage: $(basename $0) [-d DIR] [-p OPTS] [-s OPTS] [-i OPTS]

  -d, --dir DIR       working directory (default: \$PWD)
  -p, --pull OPTS     extra args appended to 'git pull'
  -s, --sync OPTS     extra args appended to 'uv sync'
  -i, --install OPTS  extra args appended to 'uv tool install'
  -h, --help          show this help
EOF
}

wd=$PWD
pull_opts=
sync_opts=
install_opts=

name=$(basename $0)
opts=$(getopt -n $name -o d:p:s:i:h -l dir:,pull:,sync:,install:,help -- "$@") ||
	{ usage >&2; exit 1; }
eval "set -- $opts"

while [ $# -gt 0 ]; do
	case $1 in
	-d|--dir) wd=$2; shift 2 ;;
	-p|--pull) pull_opts=$2; shift 2 ;;
	-s|--sync) sync_opts=$2; shift 2 ;;
	-i|--install) install_opts=$2; shift 2 ;;
	-h|--help) usage; exit 0 ;;
	--) shift; break ;;
	*) usage >&2; exit 1 ;;
	esac
done
test $# -eq 0 || { usage >&2; exit 1; }

# upgrade shortcut
test ${0##./} = $(basename $0) && wd=.. && sync_opts='-U'

for d in "$wd/lisa-packs" "$wd/lisa-hosts"; do
	if test ! -d "$d"; then
		git clone --recurse-submodules "https://github.com/lisa-harness/$(basename $d)" "$d"
		git config submodule.recurse true
		git submodule update --init --recursive
	fi
	cd "$d"
	git pull $pull_opts
done
if test -d "$wd/lisa"; then
	cd "$wd/lisa"
	git pull $pull_opts
else
    git clone 'https://github.com/lisa-harness/lisa' "$wd/lisa"
	cd "$wd/lisa"
fi
uv sync $sync_opts
cd "$wd/lisa-hosts/cli"
uv sync $sync_opts
uv tool install $install_opts hosts/cli
