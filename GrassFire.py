import copy


def GrassFire(img):
    """ Only input binary images of 0 and 255 """
    mask = copy.copy(img)

    h, w = mask.shape[:2]
    h = h-1
    w = w-1

    save_array = []
    zero_array = []
    blob_array = []
    temp_cord = []

    for y in range(h):
        for x in range(w):
            if mask.item(y, x) == 0 and x <= h:
                zero_array.append(mask.item(y, x))
            elif mask.item(y, x) == 0 and x >= w:
                zero_array.append(mask.item(y, x))

    # Looping if x == 1, and some pixels has to be burned
            while mask.item(y, x) > 0 or len(save_array) > 0:
                mask.itemset((y, x), 0)
                temp_cord.append([y, x])

                if mask.item(y - 1, x) > 0:
                    if [y - 1, x] not in save_array:
                        save_array.append([y - 1, x])

                if mask.item(y, x - 1) > 0:
                    if [y, x - 1] not in save_array:
                        save_array.append([y, x - 1])

                if mask.item(y + 1, x) > 0:
                    if [y + 1, x] not in save_array:
                        save_array.append([y + 1, x])

                if mask.item(y, x + 1) > 0:
                    if [y, x + 1] not in save_array:
                        save_array.append([y, x + 1])

                if len(save_array) > 0:
                    y, x = save_array.pop()

                else:
                    blob_array.append(temp_cord)
                    temp_cord = []
                    break

    return blob_array
