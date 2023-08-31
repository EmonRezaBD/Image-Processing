#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core/utility.hpp"
//#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"

#include <iostream>
#include<vector>
#include<string>
#include <ctype.h>
#include <fstream>
#include <tuple>

using namespace std;
using namespace cv;

//pair<bool, int> AdjustForThresholdRight(vector<vertical>Results, int Idx, int Threshold);
//pair<bool, int> AdjustForThresholdLeft(vector<vertical>Results, int Idx, int Threshold);

struct imageFrame {
    int x1, x2, y1, y2;
    cv::Mat img;
};
//"Determine the intersection of a polyline with a (image) rectangle."
struct clipRect {
        bool bDraw;
        double x1, y1, x2, y2;
    };

struct vertical {
    int k;
    double xp;
    double yp;
    double div;
    int ini;
};

pair<bool, int> AdjustForThresholdRight(vector<vertical>Results, int Idx, int Threshold);
pair<bool, int> AdjustForThresholdLeft(vector<vertical>Results, int Idx, int Threshold);

double TurtleX = 0.0; 
double TurtleY = 0.0; 

bool SetPixel(cv::Mat Img, double x, double y, cv::Scalar Color) {
    int w = Img.cols;
    int h = Img.rows;

    if (x < 0 || x >= w || y < 0 || y >= h) {
        return false;
    }
    int cx = static_cast<int>(std::round(x));
    int cy = static_cast<int>(std::round(y));

    if (cx < 0 || cx >= w || cy < 0 || cy >= h) {
        return false;
    }
    cv::Vec3b colorVec; //can be wrong
    colorVec[0] = static_cast<uchar>(Color[0]); // Blue channel
    colorVec[1] = static_cast<uchar>(Color[1]); // Green channel
    colorVec[2] = static_cast<uchar>(Color[2]); // Red channel

    Img.at<cv::Vec3b>(cy, cx) = colorVec;
    //Img.at<cv::Vec3b>(cy, cx) = Color;
    return true;

}

void Bresenham(cv::Mat Img, double x1, double y1, double x2, double y2, cv::Scalar Color) {
    double x = x1;
    double y = y1;

    double dx = abs(x2 - x1);
    double dy = abs(y2 - y1);

    if (dx == 0) {
        int ymin = static_cast<int>(std::round(std::min(y1, y2)));
        int ymax = static_cast<int>(std::round(std::max(y1, y2)));
        for (int yp = ymin; yp <= ymax; ++yp) {
            SetPixel(Img, x, yp, Color);
        }
        return;
    }

    double gradient = dy / float(dx);
    if (gradient > 1.0) {
        std::swap(dx, dy);
        std::swap(x, y);
        std::swap(x1, y1);
        std::swap(x2, y2);
    }
    double p = 2 * dy - dx;

    // Initialize the plotting points
    if (gradient > 1.0) {
        SetPixel(Img, y, x, Color);
    }
    else {
        SetPixel(Img, x, y, Color);
    }

    for (int k = 2; k <= std::round(dx) + 1; ++k) {
        if (p > 0) {
            y = (y < y2) ? y + 1 : y - 1;
            p = p + 2 * (dy - dx);
        }
        else {
            p = p + 2 * dy;
        }

        x = (x < x2) ? x + 1 : x - 1;
        if (gradient > 1.0) {
            SetPixel(Img, y, x, Color);
        }
        else {
            SetPixel(Img, x, y, Color);
        }
    }
}

void MoveTo(cv::Mat& Img, double x, double y, cv::Scalar Color) {
    TurtleX = std::round(x);
    TurtleY = std::round(y);
}

void LineTo(cv::Mat Img, double x, double y, cv::Scalar Color) {

    Bresenham(Img, TurtleX, TurtleY, std::round(x), std::round(y), Color);
    TurtleX = std::round(x);
    TurtleY = std::round(y);
}

pair<bool, double> GetPixel(cv::Mat Img, double x, double y) {
    int w = Img.size[1];
    int h = Img.size[0];

    double cx = round(x);
    double cy = round(y);
    if (cx < 0 || cx >= w || cy < 0 || cy >= h) {
        return make_pair(false, 0);
    }

   // int pixel = Img[(int)cy][(int)cx];
    cv::Vec3b pixel = Img.at<cv::Vec3b>(static_cast<int>(cy), static_cast<int>(cx));

    // Get the individual channel values (assuming it's a 3-channel BGR image)
    //int blue = pixelValue[0];
    //int green = pixelValue[1];
    //int red = pixelValue[2];
    double Value = 0.0;
    Value = Value + pixel[0];
    Value = Value + pixel[1];
    Value = Value + pixel[2];
    Value = Value / 3;

    return make_pair(true,Value);
}
//Calculates a unit vector in the direction specified by the point pair (x1, y1)-(x2, y2).
pair<double, double> GetUnityVector(double x1, double y1, double x2, double y2) {
    double ex = 0;
    double ey = 0;
    double dx = x2 - x1;
    double dy = y2 - y1;
    double len = sqrt(dx * dx + dy * dy);
    if (len == 0) {
        return std::make_pair(1, 0);
    }
    ex = dx / len;
    ey = dy / len;
    return std::make_pair(ex, ey);
}

//Rotation of a directional vector by an angle.
pair<double, double> Transform2D_Rotate(double x, double y, double AngleRad) {
    double ca = cos(AngleRad);
    double sa = sin(AngleRad);
    double san = -1 * sa;
    double dx = ca * x + san * y;
    double dy = sa * x + ca * y;
    x = dx;
    y = dy;
    return std::make_pair(x, y);
}

pair<int, int> GetSegmentLimits(vector<vertical> Results, int i) {
    int j = i;
    if (j >= Results.size()) {
        j = Results.size() - 1; 
    }
    if (j < 0) j = 0;
    int Color = Results[j].ini;
    while (j > 0) {
        if (Results[j].ini == Color) {
            j = j - 1;
        }
        else {
            j++; break;
        }
    }
    int n = i;
    while (n < Results.size() - 1) {
        if (Results[n].ini == Color) {
            n++;
        }
        else {
            n--; break;
        }
    }
    return(make_pair(j,n));
}

int GetLengthOfSegment(vector<vertical>Results, int i) {
    pair<int, int> p = GetSegmentLimits(Results, i);
    int j = p.first;
    int n = p.second;
    return n - j + 1;
}
double GetSegmentIntensityAverage(vector<vertical>Results, int i) {
    int Count = 0;
    double Sum = 0;
    int low, high;
    std::tie(low, high) = GetSegmentLimits(Results, i);
    for (int j = low; j <= high; j++) {
        Sum += Results[j].div;
        Count++;
    }
    if (Count > 0) Sum = (double) Sum / Count;
    return Sum;
}


void ListSegments(vector<vertical>Results, string Name) {
    int len = Results.size();
    int i = 0, N = 0;
    bool bPrint = true;
    if (bPrint) {
        std::cout << "------------- Segments in [" << len << "] " << Name << "-----------------" << std::endl;
    }
    //int low, high;
    int low=0;
    int high=0;
    while (i<len) {
        string sc="";
        if (Results[i].ini > 0) {
            sc = "GREEN";
        }
        else sc = "RED";
        int SegmentLen = GetLengthOfSegment(Results, i);
        
        pair<int, int>po = GetSegmentLimits(Results, i);
        low = po.first;
        high = po.second;
        int Intensity = GetSegmentIntensityAverage(Results, i);
        std::cout << "Segment [" << N << "] " << sc << " Len " << SegmentLen << " Range: " << low << " ... " << high << " Intensity: " << Intensity << std::endl;
        N = N + 1;
        i = high + 1;
    }
    cout << "------------- Segments -----------------";
}

void ClearSegment(vector<vertical> Results, int i) {
    int Color = Results[i].ini;
    pair<int, int>p = GetSegmentLimits(Results, i);
    int m = p.first; //
    while (m <= p.second) {//high
        vertical R = Results[m];
        R = {R.k, R.xp, R.yp, R.div, R.ini}; //can be wrong
        Results[m] = R;
        m = m + 1;
    }
}

pair<bool, int>GotoNextRight(vector<vertical>Results, int Index, int Co=2) {
    int Color = Co;
    if (Color == 2) {
        if (Results[Index].ini == 0) {
            Color = 1;
        }
        else Color = 0;
    }
    int Cn = Results[Index].ini;
    while (Cn != Color) {
        Index++;
        if (Index >= Results.size()) {
            return make_pair(false, 0);
        }
        Cn = Results[Index].ini;
    }
    return make_pair(true, Index);
}

std::tuple<bool, int, int>FindNextRedSegmentRight(vector<vertical>Results, int i) {

    int Color = Results[i].ini;
    int Idx = i;
    pair<bool, int>p = GotoNextRight(Results, Idx, 1); //# next green left
    if (p.first == false) {
        return make_tuple(false, 0, 0);
    }
    p = GotoNextRight(Results, Idx, 0);//# next red left
    if (p.first == false) {
        return make_tuple(false, 0, 0);
    }
    pair<bool, int> p1 = GetSegmentLimits(Results, Idx);
    return make_tuple(true, p1.first, p1.second);
}

pair<bool, int>GotoNextLeft(vector<vertical>Results, int Index, int Co = 2) {
    int Color = Co;
    if (Color == 2) {
        if (Results[Index].ini == 0) {
            Color = 1;
        }
        else Color = 0;
    }
    int Cn = Results[Index].ini;
    while (Cn != Color) {
        Index--;
        if (Index<0) {
            return make_pair(false, 0);
        }
        Cn = Results[Index].ini;
    }
    return make_pair(true, Index);
}

std::tuple<bool, int, int>FindNextRedSegmentLeft(vector<vertical>Results, int i) {
    int Color = Results[i].ini;
    int Idx = i;
    pair<bool, int>p = GotoNextLeft(Results, Idx, 0); //# next green left
    if (p.first == false) {
        return make_tuple(false, 0, 0);
    }
    p= GotoNextLeft(Results, Idx, 1);
    if (p.first == false) {
        return make_tuple(false, 0, 0);
    }
    pair<int,int> p1 = GetSegmentLimits(Results, Idx);
    return make_tuple(true, p1.first, p1.second);
}

void SetSegment(vector<vertical>Results, int Low, int High, int Color) {
    int m = Low;
    while (m <= High) {
        vertical R = Results[m];
        R = {R.k, R.xp, R.yp, R.div, Color};
        Results[m] = R;
        m = m + 1;
    }
}

std::tuple<bool, int, int>FindNextGreenSegmentLeft(vector<vertical>Results, int i) {

    int Color = Results[i].ini;
    int Idx = i;
    //bool bOk; int Idx;
    pair<bool, int> p1 = GotoNextLeft(Results, Idx, 0);// # next red left
    if (p1.first == false) return make_tuple(false, 0, 0);
    p1 = GotoNextLeft(Results, Idx, 1); // # next green left
    if (p1.first==false)return make_tuple(false, 0, 0);

    int low, high;
    std::tie(low, high) = GetSegmentLimits(Results, Idx);
    return make_tuple(true, low, high);
}

std::tuple<bool, int, int>FindNextGreenSegmentRight(vector<vertical>Results, int i) {

    int Color = Results[i].ini;
    int Idx = i;
    //bool bOk; int Idx;
    pair<bool, int> p1 = GotoNextRight(Results, Idx, 0);// # next red left
    if (p1.first == false) return make_tuple(false, 0, 0);
    p1 = GotoNextRight(Results, Idx, 1); // # next green left
    if (p1.first == false)return make_tuple(false, 0, 0);

    int low, high;
    std::tie(low, high) = GetSegmentLimits(Results, Idx);
    return make_tuple(true, low, high);

}

pair<bool, int> AdjustForThresholdLeft(vector<vertical>Results, int Idx, int Threshold) {
    if (Idx > 0) {
        if (Results[Idx].ini != Results[Idx - 1].ini) {
            return AdjustForThresholdRight(Results, Idx, Threshold);
        }
    }
    int Color = Results[Idx].ini;
    int low, high;
    std::tie(low, high) = GetSegmentLimits(Results, Idx);

    //int Index = Idx;
    low = low - 30;
    if (low < 0) low = 0;
    int Index = Idx;
    int Value1, Value2;
    while (Index > low) {
        Index--;
        Value1 = Results[Index + 1].div;
        Value2 = Results[Index].div;
        if ((Value1 - Threshold) * (Value2 - Threshold) < 0) {
            return make_pair(true, Index);
        }
    }
    return make_pair(false, Idx);
}

pair<bool, int> AdjustForThresholdRight(vector<vertical>Results, int Idx, int Threshold) {
    if (Idx < Results.size() - 1) {
        if (Results[Idx].ini != Results[Idx + 1].ini) {
            return AdjustForThresholdLeft(Results, Idx, Threshold);
        }
    }
    int Color = Results[Idx].ini;
    int low, high;
    std::tie(low, high) = GetSegmentLimits(Results, Idx);

    int Index = Idx;
    high = high + 30;
    if (high >= Results.size()) high = Results.size() - 1;
    int Value1, Value2;
    while (Index < high) {
        Index++;
        Value1 = Results[Index - 1].div;
        Value2 = Results[Index].div;
        if ((Value1 - Threshold) * (Value2 - Threshold) < 0) {
            return make_pair(true, Index);
        }
    }
    return make_pair(false, Idx);

}

double PixelDistance(vector<vertical>Results, int Index1, int Index2 ) {
    int len = Results.size();
    if (Index1 >= len)
        Index1 = len - 1;
    if (Index2 >= len)
        Index2 = len- 1;
    if (Index1 < 0)
        Index1 = 0;
    if (Index2 < 0)
        Index2 = 0;
    int x1 = Results[int(std::round(Index1))].k;
    int x2 = Results[int(std::round(Index2))].k;
    int y1 = Results[int(std::round(Index1))].xp;
    int y2 = Results[int(std::round(Index2))].xp;
    double d = std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    return d;
}




double FindVerticalProfile(cv::Mat cropped, double x0, double y0, double ScanDir) {

    double ex, ey;
    pair<double, double> p1 = GetUnityVector(0, 0, cos(ScanDir + CV_PI), sin(ScanDir + CV_PI));
    ex = p1.first;
    ey = p1.second;

    double rx, ry;
    pair<double, double> p2 = Transform2D_Rotate(ex, ey, CV_PI/2);
    rx = p2.first;
    ry = p2.second;

    int ScanLen = 1000;
    int Pixel4Average = 20;
    //Scanning in the sampling direction.
    std::vector<vertical> Results;

    for (int k = -ScanLen; k < ScanLen; k++) {
        //cout << k << " ";
        double xp = x0 + ex * k / 2.0;
        double yp = y0 + ey * k / 2.0;
        double V = 0.0;
        int Count = 0;
        int xo = -1;
        int yo = -1;

        // Scan quer
        for (int i = -Pixel4Average; i < Pixel4Average; i++) {
            double x = xp + rx * i;
            double y = yp + ry * i;

            //SetPixel(Img, x, y, (0, 0, 255));
            // Assuming there is a function GetPixel that returns a tuple (bool, double)
            std::pair<bool, double> Pixel = GetPixel(cropped, x, y);

            if (Pixel.first) {
                Count++;
                V += Pixel.second;
            }
        }

        if (Count > 0) {
            vertical v;
            v.k = k;
            v.xp = xp;
            v.yp = yp;
            v.div = V / Count;
            v.ini = 0;
            Results.push_back(v);
        }
    }
    /*int cnt = 0;
    for (auto i : Results) {
        cout <<cnt++<<" " << i.k << " " << i.xp << " " << i.yp << " " << i.div << " " << i.ini << "\n";
    }*/
    //Find Max
    int MaxValue = 0;
    int MinValue = 1000;
    int IdxMax = -1;

    for (int Idx = 0; Idx < Results.size(); ++Idx) {
        double Value = Results[Idx].div; //Value = Result[3];  

        if (Value > MaxValue) {
            MaxValue = Value;
            IdxMax = Idx;
        }

        if (Value < MinValue) {
            MinValue = Value;
            IdxMax = Idx; //May be IdMin
        }
    }

    for (int i = 0; i < Results.size(); ++i) {
        double Value = Results[i].div;
        if ((MaxValue - MinValue) > 0) {
            Value = (Value - MinValue) / (MaxValue - MinValue);
        }
        Results[i] = { Results[i].k, Results[i].xp, Results[i].yp, Value, 0 };
    }

    int coGreen = 1; //# above 1 / yy min max, > xx% distant from blackand white level
    int coRed = 0;

    int Size = Results.size(); 
    int Idx = Size;

    while (Idx>0) {
        Idx = Idx - 1;
        vertical V = Results[Idx];
        if (V.div > 0.4) {//V[3] = V.div
            double x1 = V.k;
            double y1 = V.xp;
            double x2 = x1 + rx * Pixel4Average;
            double y2 = y1 + ry * Pixel4Average;
                
            cv::Scalar color(255, 255, 255);
            MoveTo(cropped, x1, y1, color);
            LineTo(cropped, x2, y2, color);
            Idx = -1;
        }
    }
    ///# search black level
    double Blacklevel = 255.0;
    for (auto Result : Results) {
        double Value = (double)Result.div;
        if (Value < Blacklevel) Blacklevel = Value;
    }
    //# search white level
    double Whitelevel = 0.0;
    for (auto Result : Results) {
        double Value = (double)Result.div;
        if (Value> Whitelevel) Whitelevel = Value;
    }
    //# threshold : Mid between blackand white level

    double Threshold = (Blacklevel + Whitelevel) * 20 / 100.0;
    double ThresholdOuter2Black = 20 / 100.0;
    double ThresholdOuter2White = 50 / 100.0;


    double LevelDevBlack = 10.0; //# deviation to black level in percent
    double LevelDevWhite = 5.0; //# deviation to white level in percent

    for (int i = 0; i < Results.size(); i++) {
        double Value = Results[i].div;
        if (Value > Threshold) {
            Results[i] = { Results[i].k, Results[i].xp, Results[i].yp, Value, coGreen};
            double dev = std::abs(Value-Blacklevel)*100.0;
            if (dev < LevelDevBlack) {
                Results[i] = { Results[i].k, Results[i].xp, Results[i].yp, Value, coRed};
            }
            dev = std::abs(Value - Whitelevel) * 100.0;
            if (dev < Whitelevel) {
                Results[i] = { Results[i].k, Results[i].xp, Results[i].yp, Value, coRed };
            }

        }
        else {
            Results[i] = { Results[i].k, Results[i].xp, Results[i].yp, Value, coRed };
        }
    }
    int IdxLowest = 1000;
    int IdxHighest = -1000;
    for (int i = 0; i < Results.size(); i++) {
        if (i < IdxLowest) IdxLowest = i;
        if (i > IdxHighest) IdxHighest = i;
    }
    string Name = "Initial";
    ListSegments(Results,Name);
    //# remove short transitions
    bool bRemove = false;
    if (bRemove) {
        for (int i = 0; i < Results.size(); i++) {
            int Count = 0;
            int Color = Results[i].ini;
            if (Color > 0) {
                if (i > 0) {
                    if (Results[i - 1].ini != Color) {
                        int SegmentLen = GetLengthOfSegment(Results, i);
                        if (SegmentLen < 10) {
                            ClearSegment(Results, i);
                        }
                    }
                }
            }
        }
    }

    ListSegments(Results, Name = "After remove short transitions");
       
    bool bRemoveShortReds = true;
    if (bRemoveShortReds) {
        for (int i = 0; i < Results.size(); i++) {
            int Count = 0;
            int Color = Results[i].ini;
            if (Color == coRed) {
                if (i > 0) {
                    if (Results[i - 1].ini != Color) {
                        int SegmentLen = GetLengthOfSegment(Results, i);
                        if (SegmentLen < 30) {
                            bool bOK;
                            int low, high;
                            std::tie(bOK, low, high) = FindNextRedSegmentRight(Results, i);
                            if (bOK) {
                                std::tie(bOK, low, high) = FindNextRedSegmentLeft(Results, i);
                                if (bOK) {
                                    std::tie(low, high) = GetSegmentLimits(Results, i);
                                    SetSegment(Results, low, high, coGreen);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ListSegments(Results, Name = "After remove short reds");
    //# remove outer greens
    if (Results[1].ini > 0) ClearSegment(Results, 1);
    int H = Results.size() - 2;
    if(Results[H].ini > 0) ClearSegment(Results, H);

    ListSegments(Results, Name = "After remove outer greens");
    //# remove double outliers as '1111001111'

    for (int i = 0; i < Results.size(); i++) {

        if (i > (IdxLowest + 1) && (i < IdxHighest)) {
            int VMM = Results[i - 2].ini;
            int VM = Results[i - 1].ini;
            int V0 = Results[i].ini;
            int VP = Results[i + 1].ini;
            if (VM == V0) {
                if (VMM == VP && VMM != VM) {
                    vertical Result = Results[i - 1];
                    Results[i - 1] = { Result.k, Result.xp, Result.yp, Result.div, VP};
                    Result = Results[i];
                    Results[i] = { Result.k, Result.xp, Result.yp, Result.div, VP };
                }
            }
        }

    }
    //    # remove single outliers as '111101111'

    for (int i = 0; i < Results.size(); i++) {
        if (i > (IdxLowest) && (i < IdxHighest)) {
            int VM = Results[i - 1].ini;
            int V0 = Results[i].ini;
            int VP = Results[i + 1].ini;
            if (VM == VP && V0 != VM) {
                vertical Result = Results[i];
                Results[i] = { Result.k, Result.xp, Result.yp, Result.div, VP };
            }
        }

    }
    ListSegments(Results, Name = "After remove outer outliers");
    //    # combine short green with long green nearby (i.e. include the small black area)
    bool bCombine = true;
    if (bCombine) {
        for (int i = 0; i < Results.size(); i++) {
            int Color = Results[i].ini;
            if (Color>0) { //green
                if (i > 0) {
                    if (Results[i - 1].ini != Color) {
                        int SegmentLen = GetLengthOfSegment(Results, i);
                        if (SegmentLen < 10) {
                    
                            int low, high;
                            std::tie(low, high) = GetSegmentLimits(Results, i);
                            int k = low;
                            while (k <= high) {
                                int x1 = Results[k].xp;
                                int y1 = Results[k].yp;
                                double x2 = x1 + rx * Pixel4Average;
                                double y2 = y1 + ry * Pixel4Average;
                                MoveTo(cropped, x1, y1, (255, 255, 255));
                                LineTo(cropped, x2, y2, (255, 255, 255));
                                k = k + 1;
                            }
                            bool bOk; int Low, High;
                            std::tie(bOk, Low, High) = FindNextGreenSegmentLeft(Results, i);
                            if (bOk == true) {
                                if (High - Low > 100) {
                                    int Low2, High2;
                                    std::tie(Low2, High2) = GetSegmentLimits(Results, i);
                                    int GapLen = Low2-High2;
                                    if (GapLen < 20) {
                                        SetSegment(Results, Low, High2, 1);
                                    }
                                }
                            }
                            std::tie(bOk, Low, High) = FindNextGreenSegmentRight(Results, i);
                            if (bOk == true) {
                                if (High - Low > 100) {
                                    int Low2, High2;
                                    std::tie(Low2, High2) = GetSegmentLimits(Results, i);
                                    int GapLen = Low - High2;
                                    if (GapLen < 20) {
                                        SetSegment(Results, Low2, High, 1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ListSegments(Results, Name = "After combine green nearby");
    //# let only the longest green survive
    bool bOnlyLongest = true;
    if (bOnlyLongest) {
        int Longest = 0;
        for (int i = 0; i < Results.size(); i++) {
            if (Results[i].ini > 0) {
                int Len = GetLengthOfSegment(Results, i);
                if (Len > Longest) Longest = Len;
            }
        }
        if (Longest > 0) {
            for (int i = 0; i < Results.size(); i++) {
                if (Results[i].ini > 0) {
                    int Len = GetLengthOfSegment(Results, i);
                    if (Len > Longest) ClearSegment(Results, i);
                }
            }
        }
    }
    ListSegments(Results, Name = "After only longest green");
    //    # calculate the green length

    double TrimWidth = 0.0;

    for (int i = 0; i < Results.size(); i++) {
        int Count = 0;
        int Color = Results[i].ini;
        if (Color>0) { //Green
            if ((i > 0) && (i < Results.size() - 1)) {
                if (Results[i - 1].ini != Color) {
                    int low, high;
                    std::tie(low, high) = GetSegmentLimits(Results, i);
                    if ( (low > 0) &  (high < (Results.size() - 1)) ) {
                        int SegmentLen = GetLengthOfSegment(Results, i);
                        //# calculate sub-pixel threshold position
                        int xposl, xposh;
                        int Value1 = Results[low - 1].div;
                        int Value2 = Results[low].div;
                        int x1 = low - 1;
                        int y1 = Value1;
                        int x2 = low;
                        int y2 = Value2;
                        if (Value2 == Value1) xposl = low;
                        else {
                            //# towards black or white
                            int MyLevel = GetSegmentIntensityAverage(Results, low);
                            bool bOk;
                            std::tie(bOk,Idx)=GotoNextLeft(Results, low);
                            int Nextlevel = GetSegmentIntensityAverage(Results, Idx);
                            int ThresholdOuter = 0;
                            if (MyLevel > Nextlevel)
                                ThresholdOuter = ThresholdOuter2Black;
                            else
                                ThresholdOuter = ThresholdOuter2White;
                            if ( ((Value1 > ThresholdOuter) & (Value2 < ThresholdOuter)) || ((Value1 < ThresholdOuter) & (Value2 > ThresholdOuter)) ) {
                                xposl = (ThresholdOuter - Value1) / (Value2 - Value1) * (x2 - x1) + x1;
                            }
                            else {
                                pair<bool, int> p1 = AdjustForThresholdLeft(Results, low, ThresholdOuter);
                                int left = p1.second;
                                if (p1.first) {
                                    Value1 = Results[left - 1].div;//left = p1.second
                                    Value2 = Results[left].div;
                                    x1 = left - 1;
                                    y1 = Value1;
                                    x2 = left;
                                    y2 = Value2;
                                    if (abs(Value1 - Value2) < 0.001) {
                                        xposl = left;
                                    }
                                    else xposl = (ThresholdOuter - Value1) / (Value2 - Value1) * (x2 - x1) + x1;
                                }
                                else xposl = low;
                            }

                        }
                        int Value3 = Results[high].div;
                        int Value4 = Results[high + 1].div;
                        x1 = high;
                        y1 = Value3;
                        x2 = high + 1;
                        y2 = Value4;
                        if (Value2 == Value1)xposh = high;

                        else {
                            //# towards black or white
                            int MyLevel = GetSegmentIntensityAverage(Results, high);
                            bool bOk;
                            std::tie(bOk, Idx) = GotoNextRight(Results, high);
                            int Nextlevel = GetSegmentIntensityAverage(Results, Idx);
                            int ThresholdOuter = 0;
                            if (MyLevel > Nextlevel)
                                ThresholdOuter = ThresholdOuter2Black;
                            else
                                ThresholdOuter = ThresholdOuter2White;
                            if ( ((Value3 > ThresholdOuter) & (Value4 < ThresholdOuter)) || ((Value3 < ThresholdOuter) & (Value4 > ThresholdOuter)))
                                xposh = (ThresholdOuter - Value3) / (Value4 - Value3) * (x2 - x1) + x1;
                            else {
                                // # both are above or below threshold ==> adjust Index that high and high+1 cross the threshold
                                    
                                pair<bool, int> p2= AdjustForThresholdRight(Results, high, ThresholdOuter);
                                int right = p2.second;
                                if (p2.first) {
                                    Value3 = Results[right].div;
                                    Value4 = Results[right + 1].div;
                                    x1 = right;
                                    y1 = Value3;
                                    x2 = right + 1;
                                    y2 = Value4;

                                    if (abs(Value1 - Value2) < 0.001)
                                        xposh = (ThresholdOuter - Value3) / (Value4 - Value3) * (x2 - x1) + x1;
                                    else
                                        xposh = right;
                                }
                                else xposh = high;

                            }

                        }
                        //# draw the readout rect
                        int xp1 = Results[int(std::round(std::round(xposl)))].k;
                        int yp1 = Results[int(std::round(std::round(xposl)))].xp;
                        int xp2 = Results[int(std::round(std::round(xposh)))].yp;
                        int yp2 = Results[int(std::round(std::round(xposh)))].xp;

                        double xv1 = xp1 + rx * Pixel4Average;
                        double yv1 = yp1 + ry * Pixel4Average;
                        double xv2 = xp2 + rx * Pixel4Average;
                        double yv2 = yp2 + ry * Pixel4Average;

                        double xw1 = xp1 - rx * Pixel4Average;
                        double yw1 = yp1 - ry * Pixel4Average;
                        double xw2 = xp2 - rx * Pixel4Average;
                        double yw2 = yp2 - ry * Pixel4Average;

                        cv::Scalar Color1 = (0, 128, 255);
                        MoveTo(cropped, xv1, yv1, Color1);
                        LineTo(cropped, xw1, yw1, Color1);
                        LineTo(cropped, xw2, yw2, Color1);
                        LineTo(cropped, xv2, yv2, Color1);
                        LineTo(cropped, xv1, yv1, Color1);

                        //# draw the results as histogram
                        bool bFirst = true;

                        for (int j = 0; j < Results.size();j++) {

                            int i = Results[j].k;
                            int xp = Results[j].xp;
                            int yp = Results[j].yp;
                            int Value = Results[j].div;
                            int Vx = Results[j].ini;

                            SetPixel(cropped, xp, yp, (255, 255, 255));

                            double x = xp + rx * Pixel4Average;
                            double y = yp + ry * Pixel4Average;
                            SetPixel(cropped, x, y, (255, 255, 255));

                            x = xp;
                            y = yp;

                            if (Vx > 0)
                                SetPixel(cropped, x, y, (0, 0, 0)); //# black
                            else
                            SetPixel(cropped, x, y, (0, 0, 255)); //#red
                            x = xp + rx * Value * Pixel4Average;
                            y = yp + ry * Value * Pixel4Average;

                            if (bFirst == true) {
                                MoveTo(cropped, x, y, (255, 0, 255));
                                bFirst = false;
                            }
                            else LineTo(cropped, x, y, (255, 0, 255));
     
                        }
                        //# draw the width
                        //xp1 = Results[int(std::round(std::round(xposl)))].k;
                        //yp1 = Results[int(std::round(std::round(xposl)))].xp;
                        //xp2 = Results[int(std::round(std::round(xposh)))].k;
                        //yp2 = Results[int(std::round(std::round(xposh)))].xp;

                        //xv1 = xp1 + rx * 10;
                        //yv1 = yp1 + ry * 10;
                        //xv2 = xp2 + rx * 10;
                        //yv2 = yp2 + ry * 10;
                        //MoveTo(cropped, xp1, yp1, (0, 255, 0));
                        //LineTo(cropped, xv1, yv1, (0, 255, 0));
                        //MoveTo(cropped, xp2, yp2, (0, 255, 0));
                        //LineTo(cropped, xv2, yv2, (0, 255, 0));

                        //MoveTo(cropped, xp1, yp1, (0, 255, 0));
                        //LineTo(cropped, xp2, yp2, (0, 255, 0));

                        //xp1 = Results[low].k;
                        //yp1 = Results[low].xp;
                        //xp2 = Results[high].k;
                        //yp2 = Results[high].xp;

                        //string Text = string(PixelDistance(Results, xposh, xposl), 2) + "px";
                        ////(w, h), _ = cv2.getTextSize(Text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1);
                        //// cv::Size textSize;
                        //int baseline;
                        //cv::Size textSize = cv::getTextSize(Text, cv::FONT_HERSHEY_SIMPLEX, 0.3, 1, &baseline);
                        //int w = textSize.width;
                        //int h = textSize.height;

                        //double px = std::round((xp1 + xp2) / 2 - w / 2);
                        //double py = std::round((yp1 + yp2) / 2 - h / 2);
                        //cv::putText(cropped, Text,
                        //    cv::Point(int(px), int(py)),
                        //    cv::FONT_HERSHEY_SIMPLEX,
                        //    0.3,
                        //    cv::Scalar(255, 0, 0),
                        //    1,
                        //    cv::LINE_AA);
                        //TrimWidth = PixelDistance(Results, xposh, xposl);

                    }
                }
          
            }
        }

    }

    return TrimWidth;
}

//ScanDir : taking double for the radian
double FindVerticalProfileMulti(cv::Mat cropped, double x0, double y0, double ScanDir) {
   
    //std::pair<double, double> getDoublePair() {
    //    double value1 = 3.14;
    //    double value2 = 2.71;

    //    // Return a pair of double values
    //    return std::make_pair(value1, value2);
    //}

    pair<double, double> fromGetUnityVector = GetUnityVector(0, 0, cos(ScanDir + CV_PI), sin(ScanDir + CV_PI));
    double ex, ey;
    ex = fromGetUnityVector.first;
    ey = fromGetUnityVector.second;

    //(rx,ry) = (ex,ey,np.pi/2);

    pair<double, double> fromTransform2D_Rotate = Transform2D_Rotate(ex, ey, CV_PI/2);
    double rx, ry;
    rx = fromTransform2D_Rotate.first;
    ry = fromTransform2D_Rotate.second;

    int Count = 0;
    double TrimWidthSum = 0;
    int StepSize = 60;
    for (int Step = -1; Step <= 1; Step++) {
        double xc = x0 + rx * StepSize * Step;
        double yc = y0 + ry * StepSize * Step;
        double TrimWidth = FindVerticalProfile(cropped, xc, yc, ScanDir);
        if (TrimWidth > 0) {
            TrimWidthSum = TrimWidthSum + TrimWidth;
            Count = Count + 1;
        }
    }

    if (Count > 0) {
        TrimWidthSum = TrimWidthSum / Count;
    }
    return TrimWidthSum;
}




clipRect ClipToRect(double x1, double y1, double x2, double y2, double xmin, double ymin, double xmax, double ymax) {
    clipRect c1; //struct for return
    bool bSwap = false;
    
    double xdelta = xmax - xmin; //638
    double ydelta = ymax - ymin; //510

    //"sort by sequence of x" 
    if (x2 < x1) {
        double t = x1;
        x1 = x2;
        x2 = t;
        t = y1;
        y1 = y2;
        y2 = t;
        bSwap = true;
    }
    double dx = x2 - x1; //2000
    double dy = y2 - y1; //0
       
    if (x1 > xmax) {
        c1.bDraw = false;
        c1.x1 = x1;
        c1.y1 = y1;
        c1.x2 = x2;
        c1.y2 = y2;
        return c1;
    }

    if (x2 < xmin) {
        c1.bDraw = false;
        c1.x1 = x1;
        c1.y1 = y1;
        c1.x2 = x2;
        c1.y2 = y2;
        return c1;
    }
    if (x1 < xmin) {
        double t = (xmin - x1) / dx;
        x1 = xmin;
        y1 = y1 + t * dy;
        dx = x2 - x1;
        dy = y2 - y1;
    }
    if (x2 > xmax) {
        double t = (x2 - xmax) / dx; //t=0
        x2 = xmax;
        y2 = y2 - t * dy;
        dx = x2 - x1;
        dy = y2 - y1;
    }
    
    //"now x1 and x2 lie within the x-interval"
    if (y1 < ymin) {
        if (dy == 0) {
            c1.bDraw = false;
            c1.x1 = x1;
            c1.y1 = y1;
            c1.x2 = x2;
            c1.y2 = y2;
            return c1;
        }
        double t = (ymin - y1) / dy;
        if (t < 0) {
            c1.bDraw = false;
            c1.x1 = x1;
            c1.y1 = y1;
            c1.x2 = x2;
            c1.y2 = y2;
            return c1;
        }
        y1 = ymin;
        x1 = x1 + t * dx;
        dx = x2 - x1;
        dy = y2 - y1;
    }
    else {
        if (y1 > ymax) {
            if (dy == 0) {
                c1.bDraw = false;
                c1.x1 = x1;
                c1.y1 = y1;
                c1.x2 = x2;
                c1.y2 = y2;
                return c1;
            }
            double t = (y1 - ymax) / dy;
            y1 = ymax;
            x1 = x1 - t * dx;
            dx = x2 - x1;
            dy = y2 - y1;
        }
        if(y2<ymin){
            if (dy == 0) {
                c1.bDraw = false;
                c1.x1 = x1;
                c1.y1 = y1;
                c1.x2 = x2;
                c1.y2 = y2;
                return c1;
            }
            double t = (ymin - y2) / dy;
            y2 = ymin;
            x2 = x2 + t * dx;
            dx = x2 - x1;
            dy = y2 - y1;
        }
        else {
            if (y2 > ymax) {
                if (dy == 0) {
                    c1.bDraw = false;
                    c1.x1 = x1;
                    c1.y1 = y1;
                    c1.x2 = x2;
                    c1.y2 = y2;
                    return c1;
                }
                double t = (y2 - ymax) / dy;
                y2 = ymax;
                x2 = x2 - t * dx;
                dx = x2 - x1;
                dy = y2 - y1;

            }
        }
    }
    
    if (bSwap) {
        double t = x1; x1 = x2; x2 = t;
        t = y1; y1 = y2; y2 = t;
    }
    
    if ((abs(dy) > 0) || (abs(dx) > 0)) {
        //"Yes, there's something to draw!
        c1.x1 = x1;
        c1.y1 = y1;
        c1.x2 = x2;
        c1.y2 = y2;
        //return c1;
    }
    return c1;
}


imageFrame pad_img_to_fit_bbox(cv::Mat img, int x1, int x2, int y1, int y2) {
    cv::Mat updatedImg;
    cv::copyMakeBorder(img, updatedImg, -std::min(0, y1), std::max(y2 - img.rows, 0),
        -std::min(0, x1), std::max(x2 - img.cols, 0), cv::BORDER_REPLICATE);
   
    y2 += -std::min(0, y1);
    y1 += -std::min(0, y1);
    x2 += -std::min(0, x1);
    x1 += -std::min(0, x1);

    imageFrame f2;
    f2.img = updatedImg;
    f2.x1 = x1;
    f2.y1 = y1;
    f2.x2 = x2;
    f2.y2 = y2;

    //return img, x1, x2, y1, y2;
    return f2;
}

cv::Mat imcrop(cv::Mat img, vector<int> &bbox) {

    int x1 = bbox[0];
    int y1 = bbox[1];
    int x2 = bbox[2];
    int y2 = bbox[3];
    if (x1 < 0 || y1 < 0 || x2 > img.size[1] || y2 > img.size[0]) {
        imageFrame f1;
        f1 = pad_img_to_fit_bbox(img, x1, x2, y1, y2);
    }
    return img(cv::Rect(x1, y1, x2 - x1, y2 - y1)); //Can be wrong, extract sub region of image
}

void writeCSV(string filename, Mat m)
{
    ofstream myfile;
    myfile.open(filename.c_str());
    myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    myfile.close();
}

void ProcessFile(String filename) {
    //img = cv2.imread(filename + '.BMP') #start
    cv::Mat img = cv::imread(filename + ".BMP");
    
    //cv::Point start_point(100, 100);  // (x, y) coordinates of the starting point
    //cv::Point end_point(300, 400);    // (x, y) coordinates of the ending point
    //cv::Scalar color(0, 0, 255);  // Red color (BGR format)
    //int thickness = 2;
    //cv::line(img, start_point, end_point, color, thickness);
    //cv::imshow("RGB Image", img);
    //cv::waitKey(0);
    //namedWindow("OpenCV Application", W INDOW_AUTOSIZE);
   
    //cv::moveWindow("First OpenCV Application", 0, 45);
    std::cout << filename << " WxH: " << img.cols << " x " << img.rows <<"Point 1";//C
    double ReturnedTrimWidhtPixel = 0.0;

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // cv::imshow("Gray Image", gray);


    double cx = gray.size[1] / 2; //height of image
    double cy = gray.size[0] / 2; //width of image
    //std::cout <<"\n" << cx <<cy;
    int w = gray.size[1];
    int h = gray.size[0];
    int t = 0;
    int b = h - 1;

    vector<int> vec;
    vec.push_back(0);
    vec.push_back(t);
    vec.push_back(w - 1);
    vec.push_back(b);

    cv::Mat cropped = imcrop(img, vec);
    //writeCSV("Cropped Image.csv", cropped);

    cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);
   // cv::imshow("Cropped image", cropped);
    

    //Canny filter
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150, 3);


    int NLines = 0;
    int votes = 160;
    //Creating a Vec2f object representing a 2D vector
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, CV_PI / 180, votes);
    //writeCSV("Cropped Image.csv", cropped);

    // cout << "test" << endl;

    while (lines.empty())
    {
        votes = votes - 10;
        if (votes < 20)
        {
            // failure
            exit;
        }
        cv::HoughLines(edges, lines, 1, CV_PI / 180, votes);
    }
    if (lines.size() > 0) { //can be wrong
        double rho, theta;
        for (int i = 0; i < lines.size(); i++)
        {
            rho = lines[i][0];
            theta = lines[i][1];
            double a = std::cos(theta);
            double b = std::sin(theta);
            double x0 = a * rho;
            double y0 = b * rho;

            double x1 = std::round(x0 + 1000 * (-b));//-1000
            double y1 = std::round(y0 + 1000 * (a)); //320
            double x2 = std::round(x0 - 1000 * (-b));//1000
            double y2 = std::round(y0 - 1000 * (a));//320


            //double x1, y1, x2, y2;
            bool bDraw;
            clipRect C2 = ClipToRect(x1, y1, x2, y2, 0, 0, cropped.size[1] - 1, cropped.size[0] - 1);
            bDraw = C2.bDraw;
            x1 = C2.x1;
            y1 = C2.y1;
            x2 = C2.x2;
            y2 = C2.y2;

            if (bDraw) {
                NLines = NLines + 1;
                x0 = (x1 + x2) / 2; //x2=638
                y0 = (y1 + y2) / 2; //y1=y2=134

                x1 = std::round(x0 + 20 * (-b));
                y1 = std::round(y0 + 20 * (a)); //134
                x2 = std::round(x0 - 20 * (-b)); //638
                y2 = std::round(y0 - 20 * (a)); //134

                if (NLines == 1) {
                    ReturnedTrimWidhtPixel = FindVerticalProfileMulti(cropped, x0, y0, theta);
                }
            }
        }
    }
    else cout << "No lines found\n";
    std::string fn = filename + " FinalWWW=" + std::to_string(ReturnedTrimWidhtPixel) + "px" + ".png";
    cout << "Point 2 \n";
    cv::imwrite(fn, cropped);

}

int main()
{
    String sPath = "D:\\openCVProject\\";
    ProcessFile(sPath + "14264320");

    cv::waitKey(0);
    cv::destroyAllWindows();
     
    return 0;
}