
#include "multithreadcapture.h"
#include <unistd.h>

#include "FProfiler/FProfiler.h"

#include "keypoints/yape.h"
#include "viewsets/object_view.h"


static const int MAX_KEYPOINTS = 4096;

MultiThreadCaptureManager* MultiThreadCaptureManager::instance = NULL;

MultiThreadCapture* MultiThreadCaptureManager::getCaptureForCam( CvCapture* cam )
{
    MultiThreadCapture* result = 0;
    lock.Wait();
    if ( capture_map.find( cam ) != capture_map.end() )
        result = capture_map[cam];
    lock.Signal();
    return result;
}

void MultiThreadCaptureManager::addCaptureForCam(CvCapture* cam, MultiThreadCapture* cap )
{
    lock.Wait();
    capture_map[cam] = cap;
    lock.Signal();
}

void MultiThreadCaptureManager::removeCaptureForCam( CvCapture* cam )
{
    lock.Wait();
    capture_map.erase( cam );
    lock.Signal();
}

MultiThreadCapture::MultiThreadCapture( CvCapture* _capture )
: capture (_capture), FThread(), width(0), height(0), nChannels(0), frame_processsize( 0 ),
    last_frame(0),  last_frame_ret(0),  last_frame_draw(0),     last_frame_draw_ret(0),     last_frame_working(0),
    timestamp(0),   timestamp_ret(0),   timestamp_draw(0),      timestamp_draw_ret(0),      timestamp_working(0),
    processed(0),   processed_ret(0),   processed_working(0),
    capture_frame(0), process_thread_should_exit(false), process_thread_semaphore(0), /* start semaphore in busy state */
    object_input_view(0), object_input_view_ret(0), object_input_view_working(0),
    new_draw_frame_available(false),
    new_detect_frame_available(false),
    feature_detector(0)
{
    MultiThreadCaptureManager* manager = MultiThreadCaptureManager::getInstance();
    assert( manager->getCaptureForCam( capture ) == NULL );
    manager->addCaptureForCam( capture, this );

    kp_working = new keypoint[MAX_KEYPOINTS];
    kp_working_count = 0;
    kp_draw = new keypoint[MAX_KEYPOINTS];
    kp_draw_count = 0;
    kp_draw_ret = new keypoint[MAX_KEYPOINTS];
    kp_draw_ret_count = 0;
    kp_detect = new keypoint[MAX_KEYPOINTS];
    kp_detect_count = 0;
    kp_detect_ret = new keypoint[MAX_KEYPOINTS];
    kp_detect_ret_count = 0;

}

MultiThreadCapture::~MultiThreadCapture()
{
    printf("in ~MultiThreadCapture\n");
    stopCapture();

    MultiThreadCaptureManager* manager = MultiThreadCaptureManager::getInstance();
    assert( manager->getCaptureForCam( capture ) != NULL );
    manager->removeCaptureForCam( capture );

    delete[] kp_draw;
    delete[] kp_draw_ret;
    delete[] kp_detect;
    delete[] kp_detect_ret;

	if ( capture_frame )
	{
		cvReleaseImage( &capture_frame );
	}

    // release process size for frame capture, if it's unique
    if ( frame_processsize &&
        frame_processsize != last_frame &&
        frame_processsize != last_frame_ret &&
        frame_processsize != last_frame_draw &&
        frame_processsize != last_frame_draw_ret &&
        frame_processsize != last_frame_working &&
        frame_processsize != processed &&
        frame_processsize != processed_ret &&
        frame_processsize != processed_working )
    {
        cvReleaseImage( &frame_processsize );
        frame_processsize = NULL;
    }

    if ( last_frame_working )
        cvReleaseImage( &last_frame_working );

    if ( timestamp_working )
        delete timestamp_working;
    // release frame
    if ( last_frame )
    {
        cvReleaseImage( &last_frame );
        cvReleaseImage( &processed );
        if ( timestamp )
            delete timestamp;
        last_frame = NULL;
    }
    // release swapped frame
    if ( last_frame_ret )
    {
        cvReleaseImage( &last_frame_ret );
        cvReleaseImage( &processed_ret );
        delete timestamp_ret;
        last_frame_ret = NULL;
    }
    if ( last_frame_draw )
    {
        cvReleaseImage( &last_frame_draw );
        delete timestamp_draw;
    }
    if ( last_frame_draw_ret )
    {
        cvReleaseImage( &last_frame_draw_ret );
        delete timestamp_draw_ret;
    }


}


void MultiThreadCapture::startCapture()
{
    if ( thread_running )
    {
        printf("MultiThreadCapture(%x)::StartCapture(): capture already running\n", this );
        return;
    }
    new_draw_frame_available=false;
    new_detect_frame_available=false;

    // start the processing thread
    startProcessThread( 30 /* priority, only if running as root */ );
    // start the capturing trhead
    FThread::StartThread();
}

void MultiThreadCapture::stopCapture()
{
    // stop the processing thread
    stopProcessThread();

    // stop the capture thread
    FThread::StopThread();

    new_draw_frame_available=false;
    new_detect_frame_available=false;

}

void MultiThreadCapture::ThreadedFunction()
{
    // try to get the frame
    PROFILE_THIS_BLOCK("mtc thread func");
    PROFILE_SECTION_PUSH("grab");
    bool grabbed = cvGrabFrame( capture );
    PROFILE_SECTION_POP();
    if ( grabbed )
    {
        /*
        // calculate fps
        static FTime prev_time;
        FTime now;
        now.SetNow();
        static float capture_fps = 0;
        capture_fps = 1.0f/(now-prev_time).ToSeconds();
        printf("capture fps %2.2f\n", capture_fps );
        prev_time = now;*/

        PROFILE_SECTION_PUSH("retrieve");
        IplImage* f = cvRetrieveFrame( capture );
        PROFILE_SECTION_POP();
        if ( f )
        {
            // got! now put into capture frame

            PROFILE_SECTION_PUSH("mtc capture frame");

            capture_frame_lock.Wait();
            // store locally
            if ( capture_frame==0 || capture_frame->width != f->width ||
                capture_frame->height != f->height ||
                capture_frame->depth != f->depth )
            {
                if ( capture_frame )
                    cvReleaseImage( &capture_frame );
                capture_frame = cvCreateImage( cvGetSize( f ) , IPL_DEPTH_8U, f->nChannels );
            }
            cvCopy( f, capture_frame );

            capture_frame_lock.Signal();

            PROFILE_SECTION_POP();

            // tell the process thread
            //printf("signalling process_thread_semaphore\n");
            process_thread_semaphore.Signal();
        }

    }
    else
    {
        printf("cvGrabFrame failed: trying to rewind\n");
        // try rewinding
        cvSetCaptureProperty( capture, CV_CAP_PROP_POS_FRAMES, 0 );
    }

    // wait a bit
    usleep( 100 );

}


void MultiThreadCapture::startProcessThread( int thread_priority )
{
	pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
    // launch the thread
    pthread_create( &process_thread, &thread_attr, processPthreadFunc, this );
    pthread_attr_destroy( &thread_attr );

    if ( thread_priority > 0 )
    {

        printf("attempting to set process thread priority to %i\n", thread_priority );
        struct sched_param param;
        param.sched_priority = thread_priority;
        int res = pthread_setschedparam( process_thread, SCHED_RR, &param );
        if ( res != 0 )
        {
            printf("pthread_setschedparam failed: %s\n",
                   (res == ENOSYS) ? "ENOSYS" :
                   (res == EINVAL) ? "EINVAL" :
                   (res == ENOTSUP) ? "ENOTSUP" :
                   (res == EPERM) ? "EPERM" :
                   (res == ESRCH) ? "ESRCH" :
                   "???"
                   );
        }

    }
}

void MultiThreadCapture::stopProcessThread()
{
	process_thread_should_exit = true;
	process_thread_semaphore.Signal();
	void* ret;
	pthread_join( process_thread, &ret );
}

void* MultiThreadCapture::processPthreadFunc( void* _data )
{
	MultiThreadCapture* instance = (MultiThreadCapture*)_data;

	instance->processThreadedFunction();

	pthread_exit(0);
}

void MultiThreadCapture::processThreadedFunction( )
{

    //printf("entering processThreadedFunction()\n");
    while ( true )
    {
        process_thread_semaphore.Wait();
        //printf("process_thread_semaphore got the go ahead\n");

        if ( process_thread_should_exit)
            break;

        // try to get the lock
        // if we can't get it, that means there's a new frame
        // so just wait for the new frame semaphore
        if ( !capture_frame_lock.TryWait() )
        {

            //printf("capture frame lock not available: starting again\n");
            continue;
        }

        PROFILE_SECTION_PUSH("mtc process frame");
        // store timestamp
        if ( timestamp_working == 0 )
        {
            timestamp_working = new FTime();
        }
        timestamp_working->SetNow();


        // capture_frame_lock.Wait(); <-- already have the lock from above
        // store locally
        if ( last_frame_working==0 || last_frame_working->width != capture_frame->width ||
            last_frame_working->height != capture_frame->height ||
            last_frame_working->depth != capture_frame->depth )
        {
            if ( last_frame_working )
                cvReleaseImage( &last_frame_working );
            last_frame_working = cvCreateImage( cvGetSize( capture_frame ) , IPL_DEPTH_8U, capture_frame->nChannels );
        }
        cvCopy( capture_frame, last_frame_working );
        capture_frame_lock.Signal();



        // process_working
        if ( width != 0 )
        {
            PROFILE_SECTION_PUSH("processing");
            CvSize process_size = cvSize( width, height );
            CvSize frame_size = cvGetSize( last_frame_working );

            // ok
            // if framesize == processsize and nChannels correct, then processed_working=frame
            // if framesize != processsize and nChannels correct, resize frame->processed_working

            // if framesize == processsize and nChannels wrong,         framedetect=frame then cvtcolor framedetect->processed_working
            // if framesize != processsize and nChannels wrong, resize frame->framedetect then cvtcolor framedetect->processed_working

            // already processed_working?
            bool frame_process_same_size = (frame_size.width == process_size.width && frame_size.height == process_size.height);
            if ( last_frame_working->nChannels == nChannels )
            {
                if ( frame_process_same_size )
                    processed_working = last_frame_working;
                else
                {
                    PROFILE_SECTION_PUSH( "resize" );
                    if ( !processed_working )
                        processed_working = cvCreateImage( cvSize( process_size.width, process_size.height ), IPL_DEPTH_8U, nChannels );
                    cvResize( last_frame_working, processed_working );
                    PROFILE_SECTION_POP();
                }
            }
            else
            {
                PROFILE_SECTION_PUSH( "resize" );
                if ( frame_process_same_size )
                    frame_processsize = last_frame_working;
                else
                {
                    if ( !frame_processsize || frame_processsize->width != process_size.width || frame_processsize->height != process_size.height )
                    {
                        if ( frame_processsize )
                            cvReleaseImage( &frame_processsize );
                        frame_processsize = cvCreateImage( process_size, IPL_DEPTH_8U, capture_frame->nChannels );
                    }
                    cvResize( last_frame_working, frame_processsize );
                }
                PROFILE_SECTION_POP();

                PROFILE_SECTION_PUSH( "convert colors" );
                if ( !processed_working )
                    processed_working = cvCreateImage( cvSize( process_size.width, process_size.height ), IPL_DEPTH_8U, nChannels );
                int convert = (nChannels == 1?CV_RGBA2GRAY:CV_GRAY2RGBA);
                cvCvtColor( frame_processsize, processed_working, convert );
                PROFILE_SECTION_POP();
                /*PROFILE_SECTION_PUSH( "equalize hist" );
                cvEqualizeHist( processed_working, processed_working );
                PROFILE_SECTION_POP();*/
            }

            // keypoints
            if ( feature_detector )
            {
//                    feature_detector_lock.Wait();

                //check_target_size(input_image);
                //point_detector->set_use_bins(use_bins_for_input_image);
                //point_detector->set_tau(point_detector_tau);

                feature_detector_lock.Wait();
                if ( object_input_view_working == NULL )
                    object_input_view_working = new object_view(processed_working->width,
                                                        processed_working->height,
                                                        num_pyramid_levels );

                kp_working_count = feature_detector->pyramidBlurDetect(processed_working,
                                                                    kp_working, MAX_KEYPOINTS,
                                                                   &object_input_view_working->image);
                //printf("found %i points\n", kp_working_count);
                feature_detector_lock.Signal();

                // done
                //feature_detector_lock.Signal();
            }

            PROFILE_SECTION_PUSH("transfer ptr/ptr");

            // ok, now transfer to pointers
            last_frame_lock.Wait();

            // copy last_frame to last_frame_draw
            if ( last_frame_draw == NULL )
                last_frame_draw = (IplImage*)cvClone( last_frame_working );
            else
                cvCopy( last_frame_working, last_frame_draw );

            // put working pointers to real pointers
            IplImage* temp = last_frame;
            last_frame = last_frame_working;
            last_frame_working = temp;

            temp = processed;
            processed = processed_working;
            processed_working = temp;

            // timestamp
            if ( timestamp == NULL )
                timestamp = new FTime();
            *timestamp = *timestamp_working;
            if ( timestamp_draw == NULL )
                timestamp_draw = new FTime();
            // copy
            *timestamp_draw = *timestamp_working;

            // copy features
            kp_detect_count = kp_working_count;
            kp_draw_count = kp_working_count;
            for ( int i=0; i<kp_working_count;i ++ )
            {
                kp_detect[i] = kp_working[i];
                kp_draw[i] = kp_working[i];
            }

            // object view
            object_view* temp_o = object_input_view;
            object_input_view = object_input_view_working;
            object_input_view_working = object_input_view;

            // we have a new frame
            new_draw_frame_available = true;
            new_detect_frame_available = true;

            last_frame_lock.Signal();

            PROFILE_SECTION_POP();


            PROFILE_SECTION_POP(); // "processing"
        } // if ( width != 0 )

        PROFILE_SECTION_POP(); // "mtc process frame"
    } // while (true)
}


bool MultiThreadCapture::getCopyOfLastFrame( IplImage** last_frame_copy, FTime* timestamp_copy, bool block_until_available )
{
    PROFILE_THIS_FUNCTION();
    // block until new frame is available
    if ( !new_draw_frame_available )
    {
        if ( !block_until_available )
            return false;
        else
        {
            //printf("blocking until new frame available\n");
            int timeout_us = 10000*1000; // 10000ms = 10s
            PROFILE_SECTION_PUSH("blocking");
            while ( !new_draw_frame_available && timeout_us > 0 )
            {
                usleep( 1000 );
                timeout_us -= 1000;
            }
            PROFILE_SECTION_POP();
            if ( !new_draw_frame_available )
            {
                printf("MultiThreadCapture::getCopyOfLastFrame timed out waiting for new_draw_frame_available\n");
                return false;
            }
        }
    }
    // lock
    PROFILE_SECTION_PUSH("waiting")
    last_frame_lock.Wait();
    PROFILE_SECTION_POP();
    bool ret = true;
    // have a last frame?
    if ( last_frame_draw == NULL )
    {
        *last_frame_copy = NULL;
        ret = false;
    }
    else
    {
        // copy the frame into target

        // were we passed a target image pointer?
        if ( *last_frame_copy )
        {
            // check dimensions of passed in target image
            if (
                (*last_frame_copy)->width != last_frame_draw->width ||
                (*last_frame_copy)->height != last_frame_draw->height ||
                (*last_frame_copy)->nChannels != last_frame_draw->nChannels ||
                (*last_frame_copy)->depth != last_frame_draw->depth )
            {
                fprintf(stderr, "input size doesn't match: last_frame_draw is %ix%i %i %i, but we were given %ix%i %i %i",
                        last_frame_draw->width, last_frame_draw->height, last_frame_draw->depth, last_frame_draw->nChannels,
                        (*last_frame_copy)->width, (*last_frame_copy)->height, (*last_frame_copy)->depth, (*last_frame_copy)->nChannels );
                return false;
            }
        }
        // construct a new image pointer
        else /*( *last_frame_copy == NULL )*/
        {
            *last_frame_copy = cvCreateImage( cvGetSize( last_frame_draw ), last_frame_draw->depth, last_frame_draw->nChannels );
        }
        cvCopy( last_frame_draw, *last_frame_copy );

        if ( timestamp_copy )
            // invoke assignment operator=
            *timestamp_copy = *timestamp_draw;
    }
    last_frame_lock.Signal();
    return ret;
}

bool MultiThreadCapture::getLastDrawFrame(
    IplImage** rawFrame, FTime* timeStamp_copy, keypoint** keypoints, int *num_keypoints, bool block_until_available )
{
   PROFILE_THIS_FUNCTION();

    // handle frame availability
    if ( !new_draw_frame_available )
    {
        if ( block_until_available )
        {
            int timeout_us = 10000*1000; // 10000ms = 10s
            PROFILE_SECTION_PUSH("blocking");
            while ( !new_draw_frame_available && timeout_us > 0 )
            {
                usleep( 1000 );
                timeout_us -= 1000;
            }
            PROFILE_SECTION_POP();
            if ( !new_draw_frame_available )
            {
                printf("getLastCapturedFrame timed out waiting for new_draw_frame_available\n");
                return false;
            }
        }
        else
        {
            return false;
        }
    }

    if ( last_frame_draw )
    {
        PROFILE_SECTION_PUSH("waiting");
        last_frame_lock.Wait();
        PROFILE_SECTION_POP();
        // get image pointers
        if ( rawFrame )
            *rawFrame = last_frame_draw;
        if ( timeStamp_copy )
            *timeStamp_copy = *timestamp_draw;
        if ( keypoints )
        {
            *keypoints = kp_draw;
            *num_keypoints = kp_draw_count;
        }
        // swap pointers to keep processed images valid until the next call
        swapDrawPointers();
        new_draw_frame_available = false;

        last_frame_lock.Signal();
        return true;
    }
    else
        return false;

}

bool MultiThreadCapture::getLastDetectFrame(
    IplImage** processedFrame, IplImage** rawFrame, FTime* timestamp_copy,
    keypoint** keypoints, int *num_keypoints, object_view** gimme_object_input_view, bool block_until_available )
{
    // handle frame availability
    if ( !new_detect_frame_available )
    {
        if ( block_until_available )
        {
            int timeout_us = 10000*1000; // 10000ms = 10s
            while ( !new_detect_frame_available && timeout_us > 0 )
            {
                usleep( 1000 );
                timeout_us -= 1000;
            }
            if ( !new_detect_frame_available )
            {
                printf("MultiThreadCapture::getLastProcessedFrame timed out waiting for new_detect_frame_available\n");
                return false;
            }
        }
        else
        {
            return false;
        }
    }

    if ( last_frame )
    {
        last_frame_lock.Wait();
        // get image pointers
        if ( processedFrame )
            *processedFrame = processed;
        if ( rawFrame )
            *rawFrame = last_frame;
        if ( timestamp_copy )
            *timestamp_copy = *timestamp;
        if ( keypoints )
        {
            *keypoints = kp_detect;
            *num_keypoints = kp_detect_count;
        }
        if ( gimme_object_input_view )
            *gimme_object_input_view = object_input_view;

        // swap pointers to keep processed images valid until the next call
        swapDetectPointers();
        new_detect_frame_available = false;

        last_frame_lock.Signal();
        return true;
    }
    else
        return false;
}

void MultiThreadCapture::swapDetectPointers()
{
    IplImage* temp  = last_frame;
    last_frame      = last_frame_ret;
    last_frame_ret  = temp;

    temp            = processed;
    processed       = processed_ret;
    processed_ret   = processed;

    FTime* temp_t   = timestamp;
    timestamp       = timestamp_ret;
    timestamp_ret   = temp_t;

    keypoint* temp_k    = kp_detect;
    int temp_count      = kp_detect_count;
    kp_detect           = kp_detect_ret;
    kp_detect_count     = kp_detect_ret_count;
    kp_detect_ret       = temp_k;
    kp_detect_ret_count = temp_count;

    object_view* temp_o = object_input_view;
    object_input_view   = object_input_view_ret;
    object_input_view_ret = temp_o;

}

void MultiThreadCapture::swapDrawPointers()
{
    IplImage* temp      = last_frame_draw;
    last_frame_draw     = last_frame_draw_ret;
    last_frame_draw_ret = temp;

    FTime* temp_t       = timestamp_draw;
    timestamp_draw      = timestamp_draw_ret;
    timestamp_draw_ret  = temp_t;

    // swap kp_draw and kp_draw_ret
    keypoint* temp_k    = kp_draw;
    int temp_c          = kp_draw_count;
    kp_draw             = kp_draw_ret;
    kp_draw_count       = kp_draw_ret_count;
    kp_draw_ret         = temp_k;
    kp_draw_ret_count   = temp_c;


}


/*bool MultiThreadCapture::getLastProcessedFrameFeatures( keypoint* feature_storage, int feature_storage_size )
{

}*/


void MultiThreadCapture::setNumPyramidLevels( int nLevels )
{
    assert( object_input_view == NULL );
    num_pyramid_levels = nLevels;
}

