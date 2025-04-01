import torch
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import geohash

def get_coordinates(location_name, max_retries=3, timeout=10):
    # 使用Nominatim地理编码器
    geolocator = Nominatim(user_agent="geoapiExercises", timeout=timeout)

    retries = 0
    while retries < max_retries:
        try:
            # 获取地点的经纬度
            location = geolocator.geocode(location_name)
            if location:
                return (location.latitude, location.longitude)
            else:
                raise ValueError(f"无法找到地点: {location_name}")
        except GeocoderTimedOut:
            retries += 1
            print(f"请求超时，正在重试 ({retries}/{max_retries})...")
            time.sleep(1)  # 等待1秒后重试
    raise GeocoderTimedOut(f"在 {max_retries} 次重试后仍无法获取地点: {location_name}")


def location_to_tensor(location_name):
    # 获取经纬度坐标
    lat, lon = get_coordinates(location_name)

    # 将经纬度转换为PyTorch张量
    coordinates_tensor = torch.tensor([lat, lon], dtype=torch.float32)

    return coordinates_tensor


# 示例使用
# location_name = "青岛市即墨区"
# try:
#     coordinates_tensor = location_to_tensor(location_name)
#     print(f"{location_name} 的经纬度坐标张量: {coordinates_tensor}")
# except Exception as e:
#     print(f"错误: {e}")

lat = 36.302222
lon = 118.055711
geohash_features = geohash.encode(lat, lon, precision=12)

print(f"提取的地名: {geohash_features}")