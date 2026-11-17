from morethanahack.helpers.loader import load, check_alignment, print_report
from morethanahack.helpers.explorer import print_team_metadata


dataset = load()
print(f"\n{len(dataset.records)} frames loaded")
print_report(check_alignment(dataset), "dataset alignment check")

print_team_metadata(dataset)

frame_df = dataset.to_df()

# Identify frames where the ball is visible
ball = frame_df[['ball_x', 'ball_y']]
seen = ball.notna().all(axis=1)
print('ball present on %d of %d loaded frames (%.1f%%)' % (seen.sum(), 
                                                           len(seen),
                                                           100 * seen.mean()))

frame_df[(frame_df['frame_id'] > 11092) & (frame_df['frame_id'] < 90651)].to_csv('filtered_frames.csv', index=False)