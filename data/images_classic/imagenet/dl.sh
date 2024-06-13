for id in n02814860 n12267677 n02892201 n02056570 n04399382 n04562935 n03126707 n03937543 n03706229 n03837869 n02699494 n01768244
do
wget https://image-net.org/data/winter21_whole/${id}.tar
tar -xf ${id}.tar --one-top-level
done
