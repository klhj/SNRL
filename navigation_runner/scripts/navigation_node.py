#!/usr/bin/env python3
import sys
import rospy
from navigation import Navigation
import hydra
import os
sys.argv = [arg for arg in sys.argv if ":=" not in arg and not arg.startswith("__")]
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts/cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    rospy.init_node("navigation_node", anonymous=True)
    nav = Navigation(cfg)
    # nav.run()
    rospy.spin()


if __name__ == "__main__":
    main()
