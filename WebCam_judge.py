def judgement(frame):
    height, width = frame.shape[:2]

    res = [[0 for _ in range(width)] for _ in range(height)]

    for h in reversed(range(height)):
        for w in reversed(range(width)):
            red = frame[h][w][0]
            green = frame[h][w][1]
            blue = frame[h][w][2]
            if (red < 150 and blue < 150):  # 緑もしくは黒
                if (green < 120):  # black
                    res[height - h - 1][w] = 1
                elif (red * 1.1 < green and blue * 1.1 < green):  # green
                    res[height - h - 1][w] = 2

    res[0][0] = 2
    res[0][1] = 1
    return res
