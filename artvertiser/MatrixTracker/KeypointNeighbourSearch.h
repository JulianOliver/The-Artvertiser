#pragma once

#include "ofxVec2f.h"
#include "../../garfeild/keypoints/keypoint.h"
#include "../FProfiler/FTime.h"
#include "../FProfiler/FSemaphore.h"
#include <vector>
#include <set>
#include <algorithm>

/*#ifndef max
#define max(x,y) (((x)>(y))?(x):(y))
#endif
#ifndef min
#define min(x,y) (((x)<(y))?(x):(y))
#endif*/

using namespace std;

class KeypointNeighbourSearch
{
public:
    /// set the previous frame from this one. keypoints is the array of
    /// keypoints, num_keypoints is the number of keypoints, corners[] are
    /// the four corners in 2d space of the prior detected rect.
    /// stores the keypoints pointer.
    void addPrior( keypoint* keypoints, int num_keypoints, ofxVec2f corners[4], const FTime& timestamp );

    /// calculate bin indices for x/y points
    int xBinIndex( float x )
    {
        float x_rel = x-min_x;
        float x_pct = x_rel/(max_x-min_x);
        int bin = max(0,min(int(x_pct*int(x_bins.size())),int(x_bins.size()-1)));
        return bin;
    }
    int yBinIndex( float y )
    {
        float y_rel = y-min_y;
        float y_pct = y_rel/(max_y-min_y);
        int bin = max(0,min(int(y_pct*int(y_bins.size())),int(y_bins.size()-1)));
        return bin;
    }
    /// 0 = left, 1 = centre, 2 = right
    int xBinThird( float x ) { float x_offs = min(max_x, max(x-min_x, 0.0f)); x_offs/=x_bins.size();
        /* whack off int part */ x_offs -= int(x_offs);
        return x_offs*3; }
    /// 0 = top, 1 = centre, 2 = bottom
    int yBinThird( float y ) { float y_offs = min(max_y, max(y-min_y, 0.0f)); y_offs/=y_bins.size();
        /* whack off int part */ y_offs -= int(y_offs);
        return y_offs*3; }


    /// draw pixels over output, showing a probability image. pixels points at image(0,0), width_step is the
    /// step in unsigned char's between image(0,y) and image(0,y+1)
    void createProbabilityImage( unsigned char* pixels, int width, int height, int width_step,
                                 keypoint* this_frame_keypoints, int num_keypoints, const ofxVec2f& translation_test_centre, int radius );

    /// Return a probability for this translation being accurate.
    /// Lower is better, 0 is best; -1 means fail somehow.
    float getProbability( keypoint* this_keypoints, int num_keypoints, const ofxVec2f& translation_to_test );

    /// refine an initial 2d translation estimate by matching keypoints against the current stored prior
    bool refine( const ofxVec2f& initial_2d_translation_estimate, keypoint* keypoints,
                int num_keypoints, const FTime& timestmap,
                ofxVec2f& refined_2d_translation_estimate, float &final_score );

    /// push the current prior keypoints on to the back of the passed-in output vector
    void getPrior( vector<ofxVec2f> &output ) { lock(); output.insert( output.end(), prior_keypoints.begin(), prior_keypoints.end() ); unlock(); }

    /// put the current corners into the passed-in array
    void getCorners( ofxVec2f* corners ) { for ( int i=1; i<4; i++ ) corners[i] = prior_corners[i]; }


private:

    void lock() { sem.Wait(); }
    void unlock() { sem.Signal(); }

    float min_x, max_x, min_y, max_y;

    typedef vector< set< int > > KeypointBins;
    KeypointBins x_bins;
    KeypointBins y_bins;
    vector<ofxVec2f> prior_keypoints;
    ofxVec2f prior_corners[4];
    FTime prior_keypoints_timestamp;

    FSemaphore sem;

    /*/// select a prior bin set
    bool selectPriorBin( FTime& timestamp );

    typedef struct _KeypointBinset
    {
        KeypointBins x_bins;
        KeypointBins y_bins;
        vector<ofxVec3f>
    } KeypointBinset;

    KeypointBins* x_bins;
    KeypointBins* y_bins;
    vector<ofxVec3f>* prior_keypoints;

    typedef map< FTime, KeypointBinset > AllKeypointBins;
    AllKeypointBins all_bins;

    */


};
