import math
import json

def distance_3d(point1, point2):
    """
    Computes the 3D distance in meters between two points given as
    (latitude, longitude, depth), where depth is in meters below sea surface.
    """
    # Unpack points
    lat1, lon1, depth1 = point1
    lat2, lon2, depth2 = point2

    # Mean latitude in radians
    mean_lat = math.radians((lat1 + lat2) / 2.0)

    # Convert degree differences to meters
    dy = (lat2 - lat1) * 111_320           # meters per degree latitude
    dx = (lon2 - lon1) * 111_320 * math.cos(mean_lat)
    dz = depth2 - depth1                    # depth difference in meters

    # 3D Euclidean distance
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance



filename = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo.json"
with open(filename, 'r') as file:
    channel_pos_geo = json.load(file)
# Example points from the channel positions


filename_2 = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\source_tx_pos.json"
with open(filename_2, 'r') as file:
    source_tx_positions = json.load(file)



p1 = (channel_pos_geo['350'][0], channel_pos_geo['350'][1], channel_pos_geo['350'][2])  # (lat, lon, alt)
p2 = (source_tx_positions[100][0], source_tx_positions[100][1], -30)  # (lat, lon, alt)

print(f"Point 1 (Channel 350): {p1}")
print(f"Point 2 (Source TX 100): {p2}")





def main() -> None:
    d_1_2 = distance_3d(p1, p2)


    print(f"Distance p1 to p2: {d_1_2:.4f} m")
  



if __name__ == "__main__":
    main()  
    