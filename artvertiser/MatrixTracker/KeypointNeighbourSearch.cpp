#include "KeypointNeighbourSearch.h"
#include "../../starter/image/pyrimage.h"
#include <algorithm>

static const float BIN_WIDTH = 32.0f;



void KeypointNeighbourSearch::addPrior( keypoint* keypoints, int num_keypoints, ofxVec2f corners[4], const FTime& timestamp )
{
    // go through keypoints array, looking for points inside the 4 corners

    // for now just use the aabb
    min_x=corners[0].x;
    max_x=corners[0].x;
    min_y=corners[0].y;
    max_y=corners[0].y;
    for ( int i=1; i<4; i++ )
    {
        min_x = min( min_x, corners[i].x );
        max_x = max( max_x, corners[i].x );
        min_y = min( min_y, corners[i].y );
        max_y = max( max_y, corners[i].y );
    }

    // store corners locally
    for ( int i=1; i<4; i++ )
    {
        prior_corners[i] = corners[i];
    }

    lock();

    x_bins.clear();
    x_bins.resize( (max_x-min_x) / BIN_WIDTH );
    y_bins.clear();
    y_bins.resize( (max_y-min_y) / BIN_WIDTH );
    prior_keypoints.clear();

    // now search
    for ( int i=0; i<num_keypoints; i++ )
    {
        keypoint& kp = keypoints[i];
        int s = kp.scale;
        float x = PyrImage::convCoordf(kp.u, s, 0);
        float y = PyrImage::convCoordf(kp.v, s, 0);

        if ( x >= min_x && x <= max_x &&
             y >= min_y && y <= max_y )
        {
            // store the keypoint

            prior_keypoints.push_back( ofxVec2f( x,y ) );
            // bin
            int x_bin = xBinIndex(x);
            int y_bin = yBinIndex(y);
            x_bins[x_bin].insert(i);
            y_bins[y_bin].insert(i);
            // also quadrants
            int x_third = xBinThird( x );
            int y_third = yBinThird( y );
            /*// left/right ; at the moment only does + pattern, should do O
            if ( x_third == 0 && x_bin > 0 )
            {
                x_bins[x_bin-1].insert(i);
            }
            else if ( x_third == 2 && x_bin < x_bins.size()-1 )
            {
                x_bins[x_bin+1].insert(i);
            }
            if ( y_third == 0 && y_bin > 0 )
            {
                y_bins[y_bin-1].insert(i);
            }
            else if ( y_third == 2 && y_bin < y_bins.size()-1 )
            {
                y_bins[y_bin+1].insert(i);
            }*/
        }
    }

    unlock();
}

float KeypointNeighbourSearch::getProbability( keypoint* this_frame_keypoints, int num_keypoints,
                                             const ofxVec2f& translation )
{
    lock();

    if ( prior_keypoints.size()== 0 )
    {
        unlock();
        return -1;
    }

    //float DISTANCE_THRESH_SQ = DISTANCE_THRESH*DISTANCE_THRESH;
    // counter for number of new keypoints within our (transposed) roi
    int in_roi_count = 0;
    // distance accumulator for averaging
    float total_distance = 0;
    // counter for distance accumulator
    int distance_count = 0;
    /*printf("searching: min_x %3.2f, max_x %3.2f, min_y %3.2f, max_y %3.2f\n",
           min_x, max_x, min_y, max_y );*/
    for ( int i=0; i<num_keypoints; i++ )
    {
        // todo: use random selection

        // set seearch_keypoint by reverse-translating the current frame keypoint
        // the aim is to get this reverse-translated keypoint as close as possible to
        // a keypoint on the prior
        keypoint& kp = this_frame_keypoints[i];
        int s = kp.scale;
        float kp_x = PyrImage::convCoordf(kp.u, s, 0);
        float kp_y = PyrImage::convCoordf(kp.v, s, 0);

        ofxVec2f search_kp(kp_x-translation.x,
                           kp_y-translation.y );
        /*printf("incoming kp at %3.2f,%3.2f -> matching with prior at %3.2f,%3.2f\n",
               kp_x, kp_y, search_kp.x, search_kp.y );*/

        if (search_kp.x < min_x ||
            search_kp.x > max_x ||
            search_kp.y < min_y ||
            search_kp.y > max_y )
            continue;

        // quadrants included at construct-time

        int x_bin = xBinIndex(search_kp.x);
        int y_bin = yBinIndex(search_kp.y);
        /*
        printf("x bin is %i(/%i), y_bin is %i(/%i)\n", x_bin, x_bins.size(), y_bin, y_bins.size() );
        int count=0;
        for ( set<int>::iterator it = x_bins[x_bin].begin();
            it != x_bins[x_bin].end();
            ++it, ++count )
        {
            printf("%s %i", count==0?"":",", *it );
        }
        count = 0;
        printf("\n");
        for ( set<int>::iterator it = y_bins[y_bin].begin();
            it != y_bins[y_bin].end();
            ++it, ++count )
        {
            printf("%s %i", count==0?"":",", *it );
        }
        printf("\n");*/

        vector<int> matching;
        set_intersection( x_bins[x_bin].begin(), x_bins[x_bin].end(),
                          y_bins[y_bin].begin(), y_bins[y_bin].end(),
                          std::back_inserter(matching) );

        // make a score
        for ( vector<int>::iterator jt = matching.begin();
            jt != matching.end();
            ++jt )
        {
            ofxVec2f& prior = prior_keypoints[*jt];
            float dx = search_kp.x-prior.x;
            float dy = search_kp.y-prior.y;
            float distance_sq = dx*dx + dy*dy;
            total_distance += distance_sq;
        }
        distance_count += matching.size();
        ++in_roi_count;
    }

    unlock();

    // now divide to average, and also divide by roi count for a confidence figure
    if ( distance_count > 0 )
    {
        total_distance /= (float(distance_count)*float(in_roi_count));
        return total_distance;
    }
    else
        // fail
        return -1;

}


void KeypointNeighbourSearch::createProbabilityImage( unsigned char* pixels, int width, int height, int width_step,
                                                     keypoint* this_frame_keypoints, int num_keypoints, const ofxVec2f& centre, int radius )
{
    for ( int yoffs=max(0,int(centre.y-radius));
              yoffs<min(int(centre.y+radius),height-1);
              yoffs++ )
    {
        unsigned char* row = pixels+yoffs*width_step;
        for ( int xoffs=max(0,int(centre.x-radius));
                  xoffs<min(int(centre.x+radius),width-1);
                  xoffs++ )
        {
            float u = 2.0f*xoffs/width - 1;
            float v = 2.0f*yoffs/height - 1;
            float probability = getProbability( this_frame_keypoints, num_keypoints, ofxVec2f( u,v ) );
            row[xoffs] = 255-max(0,min(255,(int)probability));
        }
    }
}


bool KeypointNeighbourSearch::refine( const ofxVec2f& initial_translation_estimate, keypoint* this_frame_keypoints,
                                     int num_keypoints, const FTime& timestamp,
                                    ofxVec2f& refined_estimate, float & final_score )
{
    // get prior
    // do we have prior keypoints?
    if ( timestamp < prior_keypoints_timestamp )
        return false;

    if  ( prior_keypoints.size() == 0 )
        return false;

    // starting from initial_translation_estimate, refine to minimize
    static const float ROOT2 = sqrtf(2.0f);
    static const float INITIAL_LENGTH = 16.0f;

    float length = INITIAL_LENGTH;
    ofxVec2f deltas[8] = {
        ofxVec2f( 0, 1 ),
        ofxVec2f( ROOT2, ROOT2 ),
        ofxVec2f( 1, 0 ),
        ofxVec2f( ROOT2, -ROOT2 ),
        ofxVec2f( 0, -1 ),
        ofxVec2f( -ROOT2, -ROOT2 ),
        ofxVec2f( -1, 0 ),
        ofxVec2f( -ROOT2, ROOT2 )
        };

    // binary search
    float curr_score = getProbability( this_frame_keypoints, num_keypoints, initial_translation_estimate );
    if ( curr_score < 0 )
    {
        // fail
        return false;
    }
    //printf("initial score is %f, refining...\n", curr_score );
    ofxVec2f curr_delta;
    for ( int i=0; i<6; i++ )
    {
        // just for this level
        int best_delta = -1;
        float best_score = curr_score;
        for ( int j=0; j<7; j++ )
        {
            float score = getProbability( this_frame_keypoints, num_keypoints, initial_translation_estimate+curr_delta+length*deltas[j] );
            if ( score >= 0 && score < best_score )
            {
                // found a better score
                //printf(" . better score %f found at len %3.1f\n", score, length );
                best_delta = j;
                best_score = score;
            }
        }
        if ( best_delta != -1 )
        {
            curr_delta += length*deltas[best_delta];
            //printf(" > delta now %3.1f,%3.1f\n", curr_delta.x, curr_delta.y );
            curr_score = best_score;
        }
        // next level
        length *= 0.5f;
    }
    final_score = sqrtf(curr_score);
    refined_estimate = initial_translation_estimate+curr_delta;

    return true;
}

/*
bool selectPriorBin( const FTime& for_time )
{
    AllKeypointBins::iterator prev_binset = all_bins.lower_bound( for_time );
    //AllKeypointBins::iterator next_pose = all_bins.upper_bound( for_time );
    bool result;

    if ( found_poses.size() == 0 )
    {
        // nothing there
        result = false;
    }
    else if ( for_time == (*prev_binset).first )
    {
        // exactly on prev
        x_bins = (*prev_binset).second.first;
        y_bins = (*prev_binset).second.second;
        result = true
    }
    else if ( prev_pose == found_binset.begin() )
    {
        // we don't have prev
        result = false;
    }
    else
    {
        // step back
        --prev_binset;
        x_bins = (*prev_binset).second.first;
        y_bins = (*prev_binset).second.second;

        result = true;
    }

    return result;

}
*/
