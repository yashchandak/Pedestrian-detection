#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <sstream>

using namespace std;
using namespace cv;

static Mat image;
static Rect2d boundingBox;
static bool paused;
static bool selectObject = false;
static bool startSelection = false;

string tag = "Person";



static void onMouse( int event, int x, int y, int, void* )
{
  if( !selectObject )
  {
    switch ( event )
    {
      case EVENT_LBUTTONDOWN:
        //set origin of the bounding box
        startSelection = true;
        boundingBox.x = x;
        boundingBox.y = y;
        boundingBox.width = boundingBox.height = 0;
        break;
      case EVENT_LBUTTONUP:
        //sei with and height of the bounding box
        boundingBox.width = std::abs( x - boundingBox.x );
        boundingBox.height = std::abs( y - boundingBox.y );
        paused = false;
        selectObject = true;
        break;
      case EVENT_MOUSEMOVE:

        if( startSelection && !selectObject )
        {
          //draw the bounding box
          Mat currentFrame;
          image.copyTo( currentFrame );
          rectangle( currentFrame, Point((int) boundingBox.x, (int)boundingBox.y ), Point( x, y ), Scalar( 255, 0, 0 ), 2, 1 );
          imshow( "Tracking API", currentFrame );
        }
        break;
    }
  }
}

//
//  Hot keys: 
//   q - quit the program
//   p - pause video
//

int main( int argc, char** argv )
{
    //open the capture
    VideoCapture cap("/home/yash/C.mp4");
    if( !cap.isOpened() )
    {
      return -1;
    }

    //
    //  "MIL", "BOOSTING", "MEDIANFLOW", "TLD"
    //
    string tracker_algorithm = "MIL";
    if ( argc>1 ) tracker_algorithm = argv[1];

    Mat frame;
    paused = false;
    namedWindow( "Tracking API", 0 );
    setMouseCallback( "Tracking API", onMouse, 0 );

    Ptr<Tracker> tracker = Tracker::create( tracker_algorithm );
    if( tracker == NULL )
    {
        cout << "***Error in the instantiation of the tracker...***\n";
        return -1;
    }

    //get the first frame
	while(1)
	{
	    cap >> frame;
	    frame.copyTo( image );
	    imshow( "Tracking API", image );
	
	char c = (char) waitKey( 20 );
        if( c == 'q' || c == 27 )
            break;
	}

    bool initialized = false;
    int frameCounter = 0; 
	int tmp = 216 ;

    for ( ;; )
    {
        char c = (char) waitKey( 2 );
        if( c == 'q' || c == 27 )
            break;
        if( c == 'p' )
            paused = !paused;
		//if( c == 'b')
		//{initialized = false; selectObject = false;}

	

        if ( !paused )
        {

            if( selectObject )
            {

		            cap >> frame;
            if(frame.empty())
            {
                break;
            }
            frame.copyTo( image );
                if( !initialized )
                {
                    //initializes the tracker
                    if( !tracker->init( frame, boundingBox ) )
                    {
                    cout << "***Could not initialize tracker...***\n";
                    return -1;
                    }
                    initialized = true;
                }
                else
                {
                    //updates the tracker
                    if( tracker->update( frame, boundingBox ) )
                    {
						
                        rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );
						Rect2d bb = boundingBox;
						bb.x += 2;
						bb.y += 2;
						bb.width -= 4;
						bb.height -= 4;
						std::ostringstream oss;
						oss << tag << tmp << ".jpg";
						//cout<<oss.str();
						if (frameCounter%5 == 0)
						{imwrite(oss.str(), image(bb).clone());
						tmp++;}
                    }
                }
            }
            imshow( "Tracking API", image );
            frameCounter++;
			
			int cc = 0;		
			while(cc--);
			//while(!selectObject);	
        }
    }

    return 0;
}
