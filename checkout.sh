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

opts=$(getopt -n init.sh -o d:p:s:i:h -l dir:,pull:,sync:,install:,help -- "$@") ||
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
test ${0##./} = $(basename $0) && wd=.. && sync_opts='-U --all-extras'

test -d "$wd/lisa-packs" || gh repo clone lisa-packs "$wd/lisa-packs"
cd "$wd/lisa-packs"
git submodule update --init
git pull $pull_opts --recurse-submodules

test -d "$wd/lisa" || gh repo clone lisa "$wd/lisa"
cd "$wd/lisa"
git pull $pull_opts
uv sync $sync_opts

test -d "$wd/lisa-hosts" || gh repo clone lisa-hosts "$wd/lisa-hosts"
cd "$wd/lisa-hosts"
git submodule update --init
git pull $pull_opts --recurse-submodules

cd cli
uv sync $sync_opts
