#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
using namespace cv;
using namespace std; 
int valueof = 0;
int similarpixs = 9999;
int diffpixs = 1;
void mincut(vector<pair<int, int> >[], int,Mat&);
bool breadthFirstSearch(vector<pair<int, int> >[], int, int, int []);
int returnEdgeWeight(vector<pair<int, int> >[], int  , int  );
void assignNewWeight(vector<pair<int, int> >[] ,int ,int ,int );
void paintImage(Mat&, bool[], int);



class Pair
{
    private:
	int first, second;

    public: Pair()
	{

	}
    public: Pair(int first1, int second1)
	{
		first = first1;
		second = second1;
	}
	void setFirst(int temp_first)
	{
		first = temp_first;
	}
	void setSecond(int temp_second)
	{
		second = temp_second;
	}
	int getFirst()
	{
		return first;
	}
	int getSecond()
	{
		return second;
	}
};

void makegraph(vector<pair<int,int> > adj[], Mat &in_image)
{
	cout << "Building Graph " << endl;
	int row = in_image.rows;
	int col = in_image.cols;
	int X_co = 0, Y_co = 0;
	
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			X_co = col * i + 1 + j; //To track the current row and the x coordinate.

			if (i == 0)
			{
				if (j == 0)
				{
					Y_co = (col * i) + (j + 1) + 1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j + 1))
					{
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}
					Y_co = (col * (i + 1)) + (j)+1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i + 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
					
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

				}
				else if (j == col - 1)
				{
					Y_co = (i*col) + (j - 1) + 1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j - 1))
					{
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = ((i + 1)*col) + j + 1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i + 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

				}
				else
				{
					
					Y_co = (col*i) + 1 + j - 1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j - 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = (col*i) + 1 + j + 1;

					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j + 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = (col*(i + 1)) + 1 + j;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i + 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}
				}

			}
			else if (i == row - 1)
			{
				if (j == 0)
				{
					Y_co = (col * i) + (j + 1) + 1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j + 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = (col * (i - 1)) + (j)+1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i - 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

				}
				else if (j == col - 1)
				{
					Y_co = (i*col) + (j - 1) + 1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j - 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}



					Y_co = ((i - 1)*col) + j + 1;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i - 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

				}
				else
				{
					Y_co = (col*i) + 1 + (j - 1);
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j - 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = (col*i) + 1 + (j + 1);

					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j + 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = (col*(i - 1)) + 1 + j;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i - 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}
				}

			}
			else
			{
				if (j == 0) {
					Y_co = 1 + col * i + (j + 1);
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j + 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}


					Y_co = 1 + col * (i - 1) + j;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i - 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = 1 + col * (i + 1) + j;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i + 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}
				}


				else if (j == (col - 1)) {
					Y_co = 1 + col * i + (j - 1);
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j - 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = 1 + col * (i - 1) + j;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i - 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = 1 + col * (i + 1) + j;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i + 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}
				}
				else {
					Y_co = 1 + col * i + (j - 1);
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j - 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = 1 + col * i + (j + 1);
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i, j + 1))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = 1 + col * (i - 1) + j;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i - 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

					Y_co = 1 + col * (i + 1) + j;
					if ((int)in_image.at<uchar>(i, j) == (int)in_image.at<uchar>(i + 1, j))
					{
						
						adj[X_co].push_back(make_pair(Y_co, similarpixs));
					}
					else
					{
						
						adj[X_co].push_back(make_pair(Y_co, diffpixs));
					}

				}
			}

		}

	}
	cout << "Finished Building Graph " << endl;
}

int main(int argc, char** argv)
{
	if (argc != 4) {
		cout << "Usage: ../seg input_image initialization_file output_mask" << endl;
		return -1;
	}

	// Load the input image
	// the image should be a 3 channel image by default but we will double check that in teh seam_carving
	Mat in_image;
	in_image = imread(argv[1]/*, CV_LOAD_IMAGE_COLOR*/);

	if (!in_image.data)
	{
		cout << "Could not load input image!!!" << endl;
		return -1;
	}

	if (in_image.channels() != 3) {
		cout << "Image does not have 3 channels!!! " << in_image.depth() << endl;
		return -1;
	}

	Mat out_image = in_image.clone();

	ifstream f(argv[2]);
	if (!f) {
		cout << "Could not load initial mask file!!!" << endl;
		return -1;
	}

	int width = in_image.cols;
	int height = in_image.rows;
	int Vertrices = (width * height) + 2;
	Mat input_to_graph;
	cvtColor(in_image, input_to_graph, COLOR_BGR2GRAY);

	vector<pair<int,int> > *adj = new vector<pair<int, int> >[Vertrices];

	makegraph(adj, input_to_graph);
	int n;
	f >> n;

	// get the initil pixels
	for (int i = 0; i<n; ++i) {
		int x, y, t;
		f >> x >> y >> t;


		if (x<0 || x >= width || y<0 || y >= height) {
			cout << "In valid pixel mask!" << endl;
			return -1;
		}

		if (t == 1) {
			
			int sourcepixel = y * in_image.cols + x + 1;
			adj[sourcepixel].push_back(make_pair(Vertrices-1, 9999));

		}
		else {
			int sinkpixels = y * in_image.cols + x + 1;
			adj[0].push_back(make_pair(sinkpixels, 9999));
		}

		
	}

	
	Mat output_image(in_image.rows, in_image.cols, in_image.type());
	
	mincut(adj, Vertrices, output_image);

	out_image = output_image.clone();
	imwrite(argv[3], out_image);
	namedWindow("Original image", WINDOW_AUTOSIZE);
	namedWindow("Show Marked Pixels", WINDOW_AUTOSIZE);
	imshow("Original image", in_image);
	imshow("Show Marked Pixels", out_image);
	waitKey(0);
	return 0;
}


void mincut(vector<pair<int, int> > adj[], int Vertices, Mat& output_image )
{

	cout << "Running Algorihtm for Min-cut " << endl;
	int source = 0;
	int sink = Vertices - 1;
	int newWeight = 0;
	vector<pair<int, int> > *residual_graph = new vector<pair<int, int> >[Vertices];
	for (int i = 0; i < Vertices; i++)
	{
		for (int j = 0; j < adj[i].size(); j++)
		{
			
			residual_graph[i].push_back(make_pair(adj[i].at(j).first, adj[i].at(j).second));
		}
	}

	int *bfspath = new int[Vertices];

	while (breadthFirstSearch(residual_graph, source, sink, bfspath))
	{
		int max_flow = INT_MAX;
		
		for (int v = sink; v != source; v = bfspath[v])
		{
			int u = bfspath[v];
			//cout << u << " u" << endl;
			max_flow = min(max_flow, returnEdgeWeight(adj, u, v));
		}

		
		for (int v = sink; v != source; v = bfspath[v])
		{
			int u = bfspath[v];
			newWeight = returnEdgeWeight(residual_graph, u, v) - max_flow;
			assignNewWeight(residual_graph, u, v, newWeight);
			newWeight = returnEdgeWeight(residual_graph, v, u) + max_flow;
			assignNewWeight(residual_graph, v, u, newWeight);
		}

	}
	cout << "After Running Min_cut algorithm Doing BFS for Getting Min-cut" << endl;

	bool *visited_array = new bool[Vertices]();

	for (int i = 0; i < Vertices; i++)
	{
		visited_array[i] = false;
	}
	//	cout << sink + 1 << endl;
	queue<int> que;

	que.push(source);

	while (!que.empty())
	{
		int u = que.front();
		que.pop();

		for (int i = 0; i < residual_graph[u].size(); i++)
		{
			//cout << residual_graph[0].size();
			if (!visited_array[residual_graph[u].at(i).first] && returnEdgeWeight(residual_graph, u, residual_graph[u].at(i).first))
			{
				visited_array[residual_graph[u].at(i).first] = true;
				que.push(residual_graph[u].at(i).first);
			}
		}
		
	}

	paintImage(output_image, visited_array, Vertices);
}

void paintImage(Mat &output_image, bool visited_array[],int vertices )
{

	cout << "Displaying Image" << endl;
	Vec3b pixel;
	pixel[0] = 0;
	pixel[1] = 0;
	pixel[2] = 0;
	Vec3b pixel1;
	pixel1[0] = 255;
	pixel1[1] = 255;
	pixel1[2] = 255;

	int row, col;

	for (int i = 1; i < vertices; i++)
	{
		row = (i - 1) / output_image.cols;
		col = (i - 1) % output_image.cols;
		if (visited_array[i] == true)
		{
			output_image.at<Vec3b>(row, col) = pixel;
		}
		else
		{
			output_image.at<Vec3b>(row, col) = pixel1;
		}
	}


}
void assignNewWeight(vector<pair<int, int> > residual_graph[], int u, int v, int newWeight)
{

	for (int i = 0; i < residual_graph[u].size(); i++)
	{
		if (residual_graph[u].at(i).first == v)
		{
			residual_graph[u].at(i) = make_pair(v, newWeight);
		}
	}

}
int returnEdgeWeight(vector<pair<int, int> > adj[], int  u, int  v)
{

	for (int i = 0; i < adj[u].size(); i++)
	{
		if (adj[u].at(i).first == v)
		{
			return adj[u].at(i).second;
		}
	}

	return 0;
}


bool breadthFirstSearch(vector<pair<int, int> > residual_graph[], int source, int sink, int bfspath[])
{
	//cout << "inside bfs";
	//valueof++;
	//cout << valueof << "valueof" << endl;
	bool *visited_array = new bool[sink+1]();

	for (int i = 0; i < sink + 1; i++)
	{
		visited_array[i] = false;
	}
//	cout << sink + 1 << endl;
	queue<int> que;

	que.push(source);

	bfspath[source] = -1;

	while (!que.empty())
	{
		int u = que.front();
		que.pop();
		
		for (int i = 0; i < residual_graph[u].size(); i++)
		{
			//cout << residual_graph[0].size();
			if (!visited_array[residual_graph[u].at(i).first] && returnEdgeWeight(residual_graph,u, residual_graph[u].at(i).first))
			{
				visited_array[residual_graph[u].at(i).first] = true;
				que.push(residual_graph[u].at(i).first);
				bfspath[residual_graph[u].at(i).first] = u;
			}
		}
		
		if (visited_array[sink])
		{
			break;
		}

	}
	return visited_array[sink];
}
