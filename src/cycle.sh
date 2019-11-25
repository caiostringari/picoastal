#/bin/bash
# This is the main capture script controler

# defines where your code is located
workdir="/home/stringari/GitHub/PiCoastal/src/"
echo "Current work dir is : "$workdir

# this is where your python install in
export PATH="/opt/python/bin/:$PATH"

# get the current date
date=$(date)
datestr=$(date +'%Y%m%d_%H%M')
echo "Current date is : "$date

# your configuration file
cfg="/home/stringari/GitHub/PiCoastal/src/capture.json"
echo "Capture config file is : "$cfg

# your email configuration
email="/home/stringari/.gmail"
echo "Email config file is : "$email

# change to current work directory
cd $workdir

# current cycle log file
log="/home/stringari/logs/picoastal_"$datestr".log"
echo "Log file is : "$log

# call the capture script
script=capture.py
echo "Calling script : "$script
python3 $workdir$script -cfg $cfg > $log 2>&1
# echo $(<$log)

# call the notification
script=notify.py
attachemnt=$(tail -n 1 $log)
echo $attachemnt
echo "Calling script : "$script
python3 $workdir$script -cfg $email -log $log -a $attachemnt
