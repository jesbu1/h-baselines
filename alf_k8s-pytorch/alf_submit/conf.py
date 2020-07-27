class Config(object):
    alf_version="0.0.5"                        # which alf docker version to use
    traincli_version="3.0.6"                   # which traincli version to use
    username="jesse.zhang"                       # username on the devserver
    server_address="gpu-dev016.hogpu.cc"       # devserver address
    clusters={"algo-small": dict(appid="hKFYpMhRFM", appkey="jHIBuFrqkgzFLRHOqbOv"),
              "rl-small": dict(appid="GqxqZDZetf", appkey="SuDpYNyaSscCwHutHcpK"),
              "share-rtx": dict(appid="LRfqHkZcoP", appkey="YGBOEGwIMdCLhGxFFrSQ"),
              "share-1080ti": dict(appid="XWYOCkhYwY", appkey="pKVSFPhNgmpAqxNWTatD")}
    alf_dir="~/h-baselines"                            # alf root dir on desktop
    socialbot_dir="~/h-baselines/SocialRobotCustom"              # socialbot root dir on desktop
    work_home="test_gail"                      # working directory on devserver
