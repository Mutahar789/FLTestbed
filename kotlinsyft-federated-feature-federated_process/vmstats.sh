#!/bin/bash
#
# References
# ~~~~~~~~~~
#
#   Gnuplot
#   ~~~~~~~
#     http://www.yolinux.com/TUTORIALS/WebServerBenchmarking.html
#     http://www.groupsrv.com/computers/about179969.html
#

declare -r WINDOWS_GNUPLOT="${PROGRAMFILES}/gnuplot/binary/gnuplot.exe"

[ -z "${AWK}" ] && AWK="$(which gawk)"
[ -z "${AWK}" ] && AWK="$(which awk)"
if [ -z "${AWK}" ]; then
  echo "[ERROR] Missing awk. Please place in path or set AWK environment property to point to executable." >&2
  exit 1
fi

[ -z "${GNUPLOT}" ] && GNUPLOT="$(which gnuplot)"
[ -z "${GNUPLOT}" -a -f "${WINDOWS_GNUPLOT}" ] && GNUPLOT="${WINDOWS_GNUPLOT}"

set -o nounset
set -o errexit

function printHelp() {
  cat <<HELP
Usage: $0 [-h] [-f <filename>] [-g] [-G]

For a more descriptive help with examples read README.md.
HELP
}

# Where to write the CSV data to; required if you want to create the graphs afterwards
declare OUTPUT_FILE=""
# If to create the graphs after the stats have been collected.
declare CREATE_GRAPH=false
# If to only create the graphs without generating new data. Requires a specified output-file (here: input-file)
declare ONLY_CREATE_GRAPH=false
# If the output file generated will only exist temporarily.
declare IS_TEMP=true

while getopts ":hgGf:" Option; do
  case $Option in
    h )
      printHelp
      exit 0
      ;;
    f )
      OUTPUT_FILE=${OPTARG}
      IS_TEMP=false
      ;;
    G )
      if [ -z "${GNUPLOT}" ]; then
        echo "[ERROR] Missing Gnuplot Tool. Either place gnuplot in path or set GNUPLOT to point to executable." >&2
        exit 1
      else
        ONLY_CREATE_GRAPH=true
        CREATE_GRAPH=true
      fi
      ;;
    g )
      if [ -z "${GNUPLOT}" ]; then
        echo "[WARNING] Missing Gnuplot Tool. No graphs will be generated. Either place gnuplot in path or set GNUPLOT to point to executable." >&2
      else
        CREATE_GRAPH=true
      fi
      ;;
    * )
      echo "Unknown option." >&2
      echo ""
      printHelp
      exit 1
      ;;
  esac
done
shift $(($OPTIND - 1))

if [ -z "${OUTPUT_FILE}" ]; then
  if ${ONLY_CREATE_GRAPH}; then
    echo "[ERROR] Requested to only create graph but missing input file. Use -f option to specify the file to plot." >&2
    exit 1
  fi
  # Generate a temporary file if none got specified
  OUTPUT_FILE="$(mktemp --suffix=.csv)"
fi

declare -r GNUPLOT_PAPER="\
set size 1, 1; \
set terminal png size 1024, 768; \
set grid y"

declare -r GNUPLOT_DATA="\
set datafile separator ','; \
set xdata time; \
set timefmt '%s'; \
set format x '%H:%M:%S'"

declare -r GNUPLOT_LABELS="\
set title TITLE font 'GNUPLOT_DEFAULT_GDFONT,14'; \
set key outside center bottom horizontal box font 'GNUPLOT_DEFAULT_GDFONT,10'; \
set xlabel 'Time' offset 0,-2; \
set ylabel YLABEL offset 2,0; \
set xtics axis autofreq rotate by -90 font 'GNUPLOT_DEFAULT_GDFONT,10'"

declare -r GNUPLOT_PLOT_OPTIONS="smooth csplines with lines"

declare -ri COL_TIMESTAMP=1
declare -ri COL_DATE=2
declare -ri COL_TOD=3
declare -ri COL_PROCS_WAITING_NUMBER=4
declare -ri COL_PROCS_BLOCK_NUMBER=5
declare -ri COL_MEM_SWAP=6
declare -ri COL_MEM_FREE=7
declare -ri COL_MEM_BUFF=8
declare -ri COL_MEM_CACHE=9
declare -ri COL_SWAP_IN=10
declare -ri COL_SWAP_OUT=11
declare -ri COL_IO_BLOCKS_READ=12
declare -ri COL_IO_BLOCKS_WRITE=13
declare -ri COL_SYS_INTERRUPTS=14
declare -ri COL_SYS_CONTEXT_SWITCHES=15
declare -ri COL_CPU_USER_TIME=16
declare -ri COL_CPU_SYSTEM_TIME=17
declare -ri COL_CPU_IDLE_TIME=18
declare -ri COL_CPU_WAIT_TIME=19

declare -r COL_TIMESTAMP_LABEL="Timestamp"
declare -r COL_DATE_LABEL="Date"
declare -r COL_TOD_LABEL="Time of the Day"
declare -r COL_PROCS_WAITING_NUMBER_LABEL="Procs waiting"
declare -r COL_PROCS_BLOCK_NUMBER_LABEL="Procs blocked"
declare -r COL_MEM_SWAP_LABEL="Virtual Memory"
declare -r COL_MEM_FREE_LABEL="Idle Memory"
declare -r COL_MEM_BUFF_LABEL="Memory for Buffers"
declare -r COL_MEM_CACHE_LABEL="Memory for Cache"
declare -r COL_SWAP_IN_LABEL="Swapped Memory in/s"
declare -r COL_SWAP_OUT_LABEL="Swapped Memory out/s"
declare -r COL_IO_BLOCKS_READ_LABEL="I/O: Received blocks/s"
declare -r COL_IO_BLOCKS_WRITE_LABEL="I/O: Sent blocks/s"
declare -r COL_SYS_INTERRUPTS_LABEL="Interrupts/s"
declare -r COL_SYS_CONTEXT_SWITCHES_LABEL="Context Switches/s"
declare -r COL_CPU_USER_TIME_LABEL="CPU User Time"
declare -r COL_CPU_SYSTEM_TIME_LABEL="CPU System Time"
declare -r COL_CPU_IDLE_TIME_LABEL="CPU Idle Time"
declare -r COL_CPU_WAIT_TIME_LABEL="CPU Wait IO Time"

function plotMemory() {
  local file="${1}"

  "${GNUPLOT}" << GNUPLOT_CMDS
# The paper
${GNUPLOT_PAPER}
set yrange [0:]
# The data
${GNUPLOT_DATA}
set output "${file}.memory.png"
# Labels
TITLE = "VMStat Memory"
YLABEL = "Memory (MB)"
${GNUPLOT_LABELS}
plot "${file}" using 1:${COL_MEM_SWAP} ${GNUPLOT_PLOT_OPTIONS} title '${COL_MEM_SWAP_LABEL}', \
  "${file}" using 1:${COL_MEM_FREE} ${GNUPLOT_PLOT_OPTIONS} title '${COL_MEM_FREE_LABEL}', \
  "${file}" using 1:${COL_MEM_BUFF} ${GNUPLOT_PLOT_OPTIONS} title '${COL_MEM_BUFF_LABEL}', \
  "${file}" using 1:${COL_MEM_CACHE} ${GNUPLOT_PLOT_OPTIONS} title '${COL_MEM_CACHE_LABEL}'
exit
GNUPLOT_CMDS
  echo "Memory graph plotted to '${file}.memory.png'"
}

function plotCpu() {
  local file="${1}"

  "${GNUPLOT}" << GNUPLOT_CMDS
# The paper
${GNUPLOT_PAPER}
set yrange [0:105]
# The data
${GNUPLOT_DATA}
set output "${file}.cpu.png"
# Labels
TITLE = "VMStat CPU"
YLABEL = "CPU %"
${GNUPLOT_LABELS}
plot "${file}" using 1:${COL_CPU_USER_TIME} ${GNUPLOT_PLOT_OPTIONS} title '${COL_CPU_USER_TIME_LABEL}', \
  "${file}" using 1:${COL_CPU_SYSTEM_TIME} ${GNUPLOT_PLOT_OPTIONS} title '${COL_CPU_SYSTEM_TIME_LABEL}', \
  "${file}" using 1:${COL_CPU_IDLE_TIME} ${GNUPLOT_PLOT_OPTIONS} title '${COL_CPU_IDLE_TIME_LABEL}', \
  "${file}" using 1:${COL_CPU_WAIT_TIME} ${GNUPLOT_PLOT_OPTIONS} title '${COL_CPU_WAIT_TIME_LABEL}'
exit
GNUPLOT_CMDS
  echo "CPU graph plotted to '${file}.cpu.png'"
}

function plotIO() {
  local file="${1}"

  "${GNUPLOT}" << GNUPLOT_CMDS
# The paper
${GNUPLOT_PAPER}
set yrange [0:]
# The data
${GNUPLOT_DATA}
set output "${file}.io.png"
# Labels
TITLE = "VMStat I/O"
YLABEL = "I/O (blocks/s)"
${GNUPLOT_LABELS}
plot "${file}" using 1:${COL_IO_BLOCKS_READ} ${GNUPLOT_PLOT_OPTIONS} title '${COL_IO_BLOCKS_READ_LABEL}', \
  "${file}" using 1:${COL_IO_BLOCKS_WRITE} ${GNUPLOT_PLOT_OPTIONS} title '${COL_IO_BLOCKS_WRITE_LABEL}'
exit
GNUPLOT_CMDS
  echo "I/O graph plotted to '${file}.io.png'"
}

# The trap ensures that graphs are plotted afterwards and
# that temporary files get deleted.
function bashtrap() {
  if ${CREATE_GRAPH} || ${ONLY_CREATE_GRAPH}; then
    plotMemory "${OUTPUT_FILE}"
    plotCpu "${OUTPUT_FILE}"
    plotIO "${OUTPUT_FILE}"
  fi
  if ${IS_TEMP}; then
    echo "Cleaning temporary file at '${OUTPUT_FILE}'"
    rm -f "${OUTPUT_FILE}"
  fi
}

if ${ONLY_CREATE_GRAPH}; then
  bashtrap
else
  trap bashtrap TERM EXIT

  echo "Starting monitoring. Press Ctrl+C to abort..."
  echo ""

  VMSTAT_OPTS="-n" # display header only once
  VMSTAT_OPTS="${VMSTAT_OPTS} -S M" # Memory in Megabytes

  echo "Output goes to \"${OUTPUT_FILE}\"..."
  # adb shell vmstat ${VMSTAT_OPTS} -n 1|"${AWK}" 'BEGIN { OFS = "," } {$1=$1} NR == 1 { next; } NR == 2 { print "timestamp","date","tod", $0; next; } { print systime(), strftime("%Y-%m-%d,%H:%M:%S"), $0; fflush(); }'|tee "${OUTPUT_FILE}"
  adb shell vmstat 1 |"${AWK}" 'BEGIN { OFS = "," } {$1=$1} NR == 1 { next; } NR == 2 { print "timestamp","date","tod", $0; next; } { print systime(), strftime("%Y-%m-%d,%H:%M:%S"), $0; fflush(); }'|tee "${OUTPUT_FILE}"
fi