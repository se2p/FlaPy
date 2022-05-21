# import os
from crontab import CronTab


def test_crontab():
    with CronTab(user=True) as cron:
        job = cron.new(command='echo "This is a test to avoid unwanted crontab creation"')
        job.hours.on(20)
        job.minute.on(15)
        job.dow.on("SUN")


# def test_sudo():
#     os.system('sudo echo a')
