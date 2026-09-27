#!/bin/sh

usage() {
	cat <<EOF
usage: $(basename $0) [-d DIR] [-p OPTS] [-s OPTS] [-i OPTS]

  -d, --dir DIR       working directory 
                        default: \$PWD
  -p, --pull OPTS     extra args appended to 'git pull'
  -s, --sync OPTS     extra args appended to 'uv sync' 
                        default: -U when running as ./develop.sh
  -h, --help          show this help
EOF
}

wd=$PWD
pull_opts=
sync_opts=
install_opts=
name=$(basename $0)
opts=$(getopt -n $name -o d:p:s:h -l dir:,pull:,sync:,help -- "$@") ||
	{ usage >&2; exit 1; }
eval "set -- $opts"

while [ $# -gt 0 ]; do
	case $1 in
	-d|--dir) wd=$2; shift 2 ;;
	-p|--pull) pull_opts=$2; shift 2 ;;
	-s|--sync) sync_opts=$2; shift 2 ;;
	-h|--help) usage; exit 0 ;;
	--) shift; break ;;
	*) usage >&2; exit 1 ;;
	esac
done

test $# -eq 0 || { usage >&2; exit 1; }
# upgrade shortcut
test ${0##./} = $(basename $0) && wd=.. && sync_opts='-U'
# clone/pull/sync lisa
if test ! -d "$wd/lisa"; then
    git clone 'https://github.com/Lisa-Harness/lisa' "$wd/lisa"	
fi
cd "$wd/lisa"
git pull $pull_opts
uv sync $sync_opts
# clone/pull lisa-packs recursively
if test ! -d "$wd/lisa-packs"; then
	git config submodule.recurse true
	git clone "https://github.com/Lisa-Harness/lisa-packs" "$wd/lisa-packs"
	git submodule update --init --recursive
fi
cd "$wd/lisa-packs"
git pull $pull_opts
# clone/pull/sync lisa-cli
if test ! -d "$wd/lisa-cli"; then
	git clone "https://github.com/Lisa-Harness/lisa-cli" "$wd/lisa-cli"
fi
cd "$wd/lisa-cli"
git pull $pull_opts
uv sync $sync_opts
