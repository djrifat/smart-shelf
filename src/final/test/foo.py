import operator,json


test_list_dict = [
                    {
                        'faceRectangle': {
                            'top': 168, 
                            'left': 204, 
                            'width': 60, 
                            'height': 60 
                        }, 
                        'faceAttributes': {
                            'gender': 'female', 
                            'age': 25.0, 
                            'emotion': {
                                'anger': 0.0, 
                                'contempt': 0.0, 
                                'disgust': 0.0, 
                                'fear': 0.0, 
                                'happiness': 0.999, 
                                'neutral': 0.001, 
                                'sadness': 0.0, 
                                'surprise': 0.0
                            }
                        }
                    }, 
                    {
                        'faceRectangle': {
                            'top': 81, 
                            'left': 140, 
                            'width': 60, 
                            'height': 60
                        }, 
                        'faceAttributes': {
                            'gender': 'male', 
                            'age': 23.0, 
                            'emotion': {
                                'anger': 0.0, 
                                'contempt': 0.0, 
                                'disgust': 0.0, 
                                'fear': 0.0, 
                                'happiness': 1.0, 
                                'neutral': 0.0, 
                                'sadness': 0.0, 
                                'surprise': 0.0
                            }
                        }
                    }
                ]

def foo(old_dick, new_dick):

    '''
    for i in old_dick:
        gender = i['faceAttributes']['gender']
        age = i['faceAttributes']['age']
        emotions = i['faceAttributes']['emotion']
        mood = max(emotions.items(), key=operator.itemgetter(1))[0]

        print(gender)
        print(age)
        print(mood)
        bar = {
            'gender': gender,
            'age': age,
            'emotion': mood
        }
        #default_data.update({'item3': 3})
        new_dick.update(bar)
    '''

# test_list_dict
#test = {}
#print(test_list_dict)
#foo = dict(test_list_dict)

#new_dict = {item['faceAttributes']['gender']:item for item in test_list_dict}
#print(new_dict)

#for i in new_dict:
    #grail = test_list_dict[i]['faceAttributes']['age']
    #print(i)
    #grail = i[]
    #print(grail)
    #test["test"] = grail
    #print(test)
    #test.update({'age': grail})


#grail = test_list_dict[0]
#print(grail)
#print(test)

#print(test_list_dict)
#print("----",test)
'''
for i in test_list_dict:
    #print(i['faceAttributes']['gender'])
    #test = {}
    test['gender'] = i['faceAttributes']['gender']
    print(test)
    #print(test)
    #for x in i:
        #print (i[x])
'''
#foo(test_list_dict, test)
#print("++++",test)


#default_data['item3'] = 3




'''
{
    'gender': 'male',
    'age': 22,
    'emotion': 'happy'
}
'''

import cv2 # opencv
import numpy as np

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# set the rectangle background to white
rectangle_bgr = (255, 255, 255)
# make a black image
img = np.zeros((500, 500))
# set some text
text = "Some text in a box!"
# get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
# set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25
# make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
cv2.imshow("A box!", img)
cv2.waitKey(0)
