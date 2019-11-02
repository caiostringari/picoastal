# SCRIPT   : notify.py
# POURPOSE : Send email notifications.
# AUTHOR   : Caio Eadi Stringari

import os
import datetime

# arguments
import json
import argparse

# email
import smtplib
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText


def mail(user, dest, pswd, subject, text, attach):
    """
    Send emails with attachments.

    :param user: a valid gmail user.
    :param dest: any other valid email adress.
    :param pswd: your password. THIS IS A SECURITY BREACH. BE CAREFULL!
    :param subject: email subject.
    :param text: Image height.
    :param attach: any path pointing to a file.
    :return: none
    """
    msg = MIMEMultipart()

    msg["From"] = user
    msg["To"] = dest
    msg["Subject"] = subject

    msg.attach(MIMEText(text))

    if attach:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(open(attach, "rb").read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        'attachment; filename="%s"' % os.path.basename(attach))
        msg.attach(part)

    mailServer = smtplib.SMTP("smtp.gmail.com", 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    mailServer.login(user, pswd)
    mailServer.sendmail(user, dest, msg.as_string())
    mailServer.close()


def main():
    """Call the main program."""
    # read configuration file
    inp = args.credentials[0]
    if os.path.isfile(inp):
        with open(inp, "r") as f:
            cfg = json.load(f)
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))

    # get the credentials
    user = cfg["credentials"]["login"]
    dest = cfg["credentials"]["destination"]
    pswd = cfg["credentials"]["password"]

    # create a subject
    now = datetime.datetime.now()
    subject = "PiCoastal Notification - {}".format(
        now.strftime("%d/%m/%Y : %H%M"))

    # log text
    if cfg["options"]["send_log"]:
        try:
            if args.log[0]:
                body = open(args.log[0], "r").readlines()
                text = " ".join(line for line in body)
            else:
                body = "PiCoastal notification. Something went wrong!"
                print("No such file or directory {}.".format(args.log[0]))
        except Exception:
            attach = False
            body = "PiCoastal notification. Something went wrong!"
            raise IOError(
                "Error reading file or directory {}.".format(args.log[0]))

    # attachemnt
    if cfg["options"]["send_last_frame"]:
        try:
            if os.path.isfile(args.attach[0]):
                attach = args.attach[0]
            else:
                text = "PiCoastal notification. Could not find the last frame."
                attach = False
                print("No such file or directory {}.".format(args.attach[0]))
        except Exception:
            attach = False
            text = "PiCoastal notification. Something went wrong!"
            raise IOError(
                "Error reading file or directory {}.".format(args.attach[0]))

    # fire
    mail(user, dest, pswd, subject, text, attach)


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--credentials", "-cfg", "-i",
                        nargs=1,
                        action="store",
                        dest="credentials",
                        required=True,
                        help="Email configuration JSON file.",)
    parser.add_argument("--log", "-log", "-l",
                        nargs=1,
                        action="store",
                        dest="log",
                        default=["capture.log", ],
                        required=False,
                        help="Log file.",)
    parser.add_argument("--attachment", "-attachment", "-a",
                        nargs=1,
                        action="store",
                        dest="attach",
                        default=["frame.jpg", ],
                        required=False,
                        help="Attachement file.",)

    args = parser.parse_args()

    main()
