'''
Multimodal Soccer Video Summarisation System - Download Soccernet dataset
Downloads the annotations and match halves in /data/SoccerNet/
'''
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="data/SoccerNet")
# only the validation set is ran
mySoccerNetDownloader.password = "REQUIRES_PERMISSION" # once permission is obtained, input the password here to process with the main script.
mySoccerNetDownloader.downloadGames(
    files=["Labels-v2.json"],
    split=["valid"]
)
mySoccerNetDownloader.downloadGames(
    files=["1_224p.mkv", "2_224p.mkv"],
    split=["valid"]
)