# Virtual Camera with Background replacement

## Setup

```
sudo apt-get install v4l2loopback-dkms
sudo modprobe v4l2loopback
```

```
pip install -r requirements.txt
```

## Run

- Start virtual camera

```
python virtual-camera-background.py
```

- Preview result

```
python camera-preview.py --device 2
```

## License

MIT
