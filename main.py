import xml.etree.ElementTree as et
import pandas as pd
from math import radians, degrees, sin, cos, asin, acos, sqrt, pi, atan2
import numpy as np

pd.set_option('display.max_columns', None)


def extract_gpx_data(filename):
    xtree = et.parse(filename)
    xroot = xtree.getroot()

    trkseg_elem = xroot[1][2]

    output_arr = []

    for trkpt in trkseg_elem:
        lat = trkpt.attrib.get("lat")
        lon = trkpt.attrib.get("lon")

        elev = trkpt[0].text
        t = trkpt[1].text

        output_arr.append((lat, lon, elev, t))

    output_df = pd.DataFrame(output_arr, columns=['Latitude', 'Longitude', 'Elevation', 'Time'])

    return output_df


def generate_distances(lon1, lat1, lon2, lat2):
    lon1 = float(lon1)
    lon2 = float(lon2)
    lat1 = float(lat1)
    lat2 = float(lat2)

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    return 6371 * (
        acos(sin(float(lat1)) * sin(float(lat2)) + cos(float(lat1)) * cos(float(lat2)) * cos(float(lon1) - float(lon2)))
    ) * 1000


def calculate_angles(lat_diff, long_diff):
    myradians = atan2(lat_diff, long_diff)
    mydegrees = degrees(myradians)

    return mydegrees


def populate_delta_fields(df):
    df['Latitude'].apply(lambda x: float(x))
    df['Longitude'].apply(lambda x: float(x))
    df['lat_old'] = df['Latitude'].astype(float).shift(-1)
    df['lon_old'] = df['Longitude'].astype(float).shift(-1)
    df['lat_diff'] = df['lat_old'].astype(float) - df['Latitude'].astype(float)
    df['lon_diff'] = df['lon_old'].astype(float) - df['Longitude'].astype(float)

    return df

    # def cumsum_shift(s, shift=1, init_values=[0]):
    #     s_cumsum = pd.Series(np.zeros(len(s)))
    #     for i in range(shift):
    #         s_cumsum.iloc[i] = init_values[i]
    #     for i in range(shift, len(s)):
    #         s_cumsum.iloc[i] = s_cumsum.iloc[i - shift] + s.iloc[i]
    #     return s_cumsum


if __name__ == "__main__":
    loaded_df = extract_gpx_data("Fasted_10_k_no_fun.xml")
    initial_df = populate_delta_fields(loaded_df)

    # Generate the distances and the angle being moved at between records & then append them to the dataframe
    delta_dist, angle_list = [], []
    for index, row in initial_df.iterrows():
        distance_moved = generate_distances(row['Longitude'], row['Latitude'], row['lon_old'], row['lat_old'])
        delta_dist.append(distance_moved)

        angle_moved = calculate_angles(row['lat_diff'], row['lon_diff'])
        angle_list.append(angle_moved)

    # Move those arrays into data frame columns. Modify negative angle values to become positive.
    initial_df['distance_moved'] = delta_dist
    initial_df['angles_moved'] = angle_list
    initial_df['angles_moved'] %= 360

    # The Avg for back and forwards rolling needs to be weighted based on the distance travelled in that record
    # Back Rolling and Forward Rolling are utilized for finding the general direction of travel
    initial_df['back_rolling'] = (initial_df['angles_moved'] * initial_df['distance_moved']).rolling(5).sum()\
                                 / initial_df['distance_moved'].rolling(5).sum()

    initial_df['forward_rolling'] = ((initial_df['angles_moved'].iloc[::-1] * initial_df['distance_moved'].iloc[::-1])
                                     .rolling(5).sum().iloc[::-1] / initial_df['distance_moved'].iloc[::-1]
                                     .rolling(5).sum().iloc[::-1])

    initial_df['avg_rolling'] = (initial_df['back_rolling'] + initial_df['forward_rolling']) / 2
    initial_df['delta_from_tangent'] = initial_df['angles_moved'] - initial_df['avg_rolling']

    # @ TODO: Find any instances in which it flicks from -35 to +35 (i.e.) a -ve to +ve (or v.v) flip > 70* from the avg

    # Once these are found, highlight them as a 1.
    col = 'check'
    conditions = [initial_df['delta_from_tangent'] - initial_df['delta_from_tangent'].shift(1) > 70,
                  (initial_df['delta_from_tangent'] - initial_df['delta_from_tangent'].shift(1) < 70) &
                  (initial_df['delta_from_tangent'] - initial_df['delta_from_tangent'].shift(1) > 0),
                  initial_df['delta_from_tangent'] - initial_df['delta_from_tangent'].shift(1) < -70,
                  (initial_df['delta_from_tangent'] - initial_df['delta_from_tangent'].shift(1) > -70) &
                  (initial_df['delta_from_tangent'] - initial_df['delta_from_tangent'].shift(1) <= 0)
                  ]
    choices = [1, 0, 1, 0]

    initial_df["check"] = np.select(conditions, choices, default=np.nan)

    # For any 1s, go back to the nearest 0 up, and then up one more!
    # work out the distance and angle between your 1 and the 0 node.
    # Delete the preceding record before the 1.
    # Modify the 1 record with the calculated distance and angle between the 1 node and the "nearest 0 up, and then
    # up one more!" Node



    print(initial_df.head(400))
