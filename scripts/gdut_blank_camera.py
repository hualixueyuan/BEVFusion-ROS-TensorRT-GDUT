#!/usr/bin/python3

"""Publish a timestamp-synchronized transparent image for GDUT's missing CAM_BACK."""

import rospy
from sensor_msgs.msg import Image


class BlankCameraPublisher:
    def __init__(self):
        self.source_topic = rospy.get_param("~source_topic", "/cam_front/image_raw")
        self.output_topic = rospy.get_param(
            "~output_topic", "/bevfusion_gdut/cam_back/image_raw"
        )
        self.frame_id = rospy.get_param("~frame_id", "cam_back_optical_frame")
        self.encoding = rospy.get_param("~encoding", "bgra8")
        if self.encoding not in ("bgra8", "bgr8"):
            raise ValueError("encoding must be 'bgra8' or 'bgr8'")

        self.channels = 4 if self.encoding == "bgra8" else 3
        self.cached_shape = None
        self.cached_data = b""
        self.publisher = rospy.Publisher(self.output_topic, Image, queue_size=2)
        self.subscriber = rospy.Subscriber(
            self.source_topic,
            Image,
            self.callback,
            queue_size=2,
            buff_size=16 * 1024 * 1024,
            tcp_nodelay=True,
        )
        rospy.loginfo(
            "GDUT blank CAM_BACK: %s -> %s (%s, source timestamp)",
            self.source_topic,
            self.output_topic,
            self.encoding,
        )

    def callback(self, source):
        shape = (source.height, source.width, self.channels)
        if shape != self.cached_shape:
            self.cached_shape = shape
            self.cached_data = bytes(source.height * source.width * self.channels)
            rospy.loginfo(
                "GDUT blank CAM_BACK image initialized: %dx%d %s",
                source.width,
                source.height,
                self.encoding,
            )

        blank = Image()
        blank.header.seq = source.header.seq
        blank.header.stamp = source.header.stamp
        blank.header.frame_id = self.frame_id
        blank.height = source.height
        blank.width = source.width
        blank.encoding = self.encoding
        blank.is_bigendian = 0
        blank.step = source.width * self.channels
        blank.data = self.cached_data
        self.publisher.publish(blank)


if __name__ == "__main__":
    rospy.init_node("gdut_blank_cam_back")
    BlankCameraPublisher()
    rospy.spin()
