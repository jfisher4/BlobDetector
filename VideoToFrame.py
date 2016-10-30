import cv2

vidcap = cv2.VideoCapture(0)

countN = 0;
countP = 0;
while True:
    ret,image = vidcap.read()
         # save frame as JPEG file
    cv2.imshow("frame",image)





    k = raw_input("enter n for negative and p for positive, q to quit: ")
    #print "you entered", var


    #k = cv2.waitKey(1) & 0xFF
    if k == "n":#ord('n'):
        print("Saving Negative...")
        cv2.imwrite("frame%05dN.jpg" % countN, image)
        countN += 1
        
    elif k == "p":#ord('p'):
        print("Saving Positive...")
        cv2.imwrite("frame%05dP.jpg" % countP, image)
    	countP += 1
    elif k == "q":#ord('q'):
        break		
vidcap.release()
cv2.destroyAllWindows()

# Line 11 is where I am having an issue..I don't know
# why the code for escape on MacOS (53) isn't working
# so I am having to control/break to stop the script. But 
# run the script as is and use the escape key to stop it.
# If that doesn't work, control/break. 



   
