from conf import Config
import sys
import os
import subprocess


def cmd(args):
    subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stdout)


def server():
    return '{}@{}'.format(Config.username, Config.server_address)


def rsync(src, target, excludes=[]):
    def _exclude(ex):
        return "--exclude=%s" % ex

    args = ['rsync', '-rI']
    args += list(map(_exclude, excludes))
    args += [src, "%s:%s" % (server(), target)]
    cmd(args)


if __name__ == "__main__":
    """For command line arguments, see ``submit.py`` in the same directory."""
    cmd(['ssh', server(), 'mkdir', '-p', Config.work_home])

    print("Copy local directory to the server ...")
    rsync(os.path.abspath(os.path.dirname(__file__)),
          Config.work_home,
          excludes=["*.pyc", ".git"])

    print("Sync ALF code ...")
    rsync(os.path.expanduser(Config.alf_dir),
          os.path.join(Config.work_home, "alf_submit/gail/job"),
          excludes=[".git", "docs", "*.pyc", "*.gif", "*.png", "*data*", "eval_socialbot", "fetch_evals", "data"])

    print("Sync SocialRobot code ...")
    rsync(os.path.expanduser(Config.socialbot_dir),
          os.path.join(Config.work_home, "alf_submit/gail/job"),
          excludes=[".git", "pygazebo", "build", "pybind11", "*.pyc", "media"])

    print("Set up and submit ...")
    cmd(['ssh', server(), 'cd',
         os.path.join(Config.work_home, "alf_submit"),
         '&&', 'python', 'submit_hiro.py'] + sys.argv[1:])

