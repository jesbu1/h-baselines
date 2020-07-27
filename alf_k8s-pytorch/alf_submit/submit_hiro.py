from conf import Config
import argparse
import sys
import os
import subprocess
import json


# set HDFS env variables
os.environ["HADOOP_HOME"] = "/usr/local/hadoop-2.7.2"
os.environ["PATH"] = os.path.expandvars("${HADOOP_HOME}/bin/:") + os.environ["PATH"]
os.environ["HADOOP_PREFIX"] = os.path.expandvars("${HADOOP_HOME}")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-1.7.0"
os.environ["CLASSPATH"] = os.path.expandvars("$($HADOOP_HOME/bin/hadoop classpath --glob)")

# traincli url
traincli_url = "https://gallery.hobot.cc/download/algorithmplatform/traincli/project/release/linux/x86_64/general/basic/%s/traincli-%s"


def check_traincli():
    """The new devserver has always traincli update-to-date. So this function
    is no longer needed.
    """
    ver = Config.traincli_version
    binary = "traincli-" + ver
    if not os.path.isfile(binary):
        os.system("wget " + traincli_url % (ver, ver))
        os.system("chmod +x " + binary)


def _generate_job_yaml(options):
    with open('gail/job.yaml', 'r') as f:
        job_yaml = f.read()

    gpu_n = 4
    if options.search_config != '':
        with open(os.path.join("gail/job/alf/alf/examples",
                               options.search_config)) as f:
            conf = json.load(f)
            if "gpus" in conf:
                gpu_n = len(conf["gpus"])
    print(gpu_n)
    job_yaml = job_yaml.replace("__job_name__", options.job_name)
    job_yaml = job_yaml.replace("__gpu_per_worker__", str(gpu_n))
    job_yaml = job_yaml.replace("__alf_version__", Config.alf_version)
    with open("gail/job.yaml", 'w') as f:
        f.write(job_yaml)


def _generate_job_script(options):
    if options.search_config == '':
        job_path = 'gail/job/job.sh'
    else:
        job_path = 'gail/job/grid_search.sh'
    with open(job_path, 'r') as f:
        job_script = f.read()
    with open("gail/job/job.sh", 'w') as f:
        f.write(job_script)


def generate_job_files(options):
    _generate_job_yaml(options)
    _generate_job_script(options)


def choose_cluster(options):
    with open('gpucluster.yaml', 'r') as f:
        cluster_str = f.read()
    assert options.cluster in Config.clusters, \
        "Cluster name {} is unrecognized!".format(options.cluster)
    id_and_key = Config.clusters[options.cluster]
    cluster_str = cluster_str.replace("__appid__", id_and_key["appid"])
    cluster_str = cluster_str.replace("__appkey__", id_and_key["appkey"])
    with open("gpucluster.yaml", 'w') as f:
        f.write(cluster_str)

    os.system("mkdir -p $HOME/.hobot; cp gpucluster.yaml $HOME/.hobot/")


def submit():
    os.chdir("gail")
    #check_traincli()
    os.system("traincli submit -f job.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job_name", help="The job name")
    parser.add_argument("-s", "--search_config", type=str, default='',
                        help="The grid search json file")
    parser.add_argument("-c", "--cluster", default="algo-small",
                        help="The cluster to put jobs on: algo-small|rl-small|share-rtx")
    options = parser.parse_args(sys.argv[1:])


    generate_job_files(options)
    choose_cluster(options)
    submit()
