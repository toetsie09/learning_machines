FROM ros:melodic

RUN apt-get update -y && apt-get install -y ros-melodic-compressed-image-transport
RUN apt-get update -y && apt-get install -y python-pip
RUN pip install tensorflow keras
WORKDIR /root/projects/
RUN echo 'catkin_make install && source /root/projects/devel/setup.bash' >> /root/.bashrc
ADD ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt
RUN echo "----------READY for USE but this wont be printed :P----------"