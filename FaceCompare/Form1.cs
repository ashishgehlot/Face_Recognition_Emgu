using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Emgu.CV.CvEnum;
using Emgu.CV.UI;
using System.IO;
using System.Text.RegularExpressions;

namespace FaceCompare
{
    public partial class Form1 : Form
    {
        private Emgu.CV.Capture cap;
        List<Image<Gray, byte>> imageparts = new List<Image<Gray, byte>>();
        string faceRecordFolder = Environment.CurrentDirectory + "\\FaceRecords\\";
        private Dictionary<string, Image<Gray, Byte>> facesAvailable = new Dictionary<string, Image<Gray, byte>>();
        HaarCascade _cascadeClassifier = new HaarCascade("haarcascades\\haarcascade_frontalface_alt.xml");

        public Form1()
        {
            InitializeComponent();
            cap = new Emgu.CV.Capture(0);
            LoadImageData();
            StartScanning();
        }

        private void LoadImageData()
        {
            var files = Directory.EnumerateFiles(faceRecordFolder, "*.*", SearchOption.TopDirectoryOnly)
                        .Where(s => Path.GetExtension(s).ToLowerInvariant() == ".bmp").ToList();

            facesAvailable.Clear();

            files.ForEach(filePath =>
            {
                facesAvailable.Add(Path.GetFileName(filePath), new Image<Gray, byte>(filePath));
            });
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                pictureBox1.ImageLocation = openFileDialog1.FileName;
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Image<Gray, byte> img = new Image<Gray, byte>(pictureBox1.ImageLocation);

            MCvAvgComp[][] rectangles = img.DetectHaarCascade(_cascadeClassifier);

            foreach (var ract in rectangles[0])
            {
                img.ROI = ract.rect;
                imageparts.Add(img.Copy());
            }

            int x = 0;
            Directory.CreateDirectory(faceRecordFolder);
            foreach (var imgs in imageparts)
            {
                var imgss = imgs.Resize(100, 100, INTER.CV_INTER_CUBIC);
                imgss.Save(faceRecordFolder + "\\" + textBox1.Text + "(" + x + ").bmp");
                x++;
                if (x > 10) break;
            };
        }

        private void StartScanning()
        {
            Application.Idle += new EventHandler(delegate (object senders, EventArgs ee)
            {
                var mat = cap.QueryFrame();

                Bitmap bitmap = mat.Bitmap;

                Image<Gray, byte> img = new Image<Gray, byte>(bitmap);

                MCvAvgComp[][] rectangles = img.DetectHaarCascade(_cascadeClassifier, 1.2, 0, HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new System.Drawing.Size(20, 20));

                foreach (var rect in rectangles[0])
                {
                    mat.Draw(rect.rect, new Bgr(255, 0, 0), 2);
                    img = new Image<Gray, byte>(bitmap);
                    string name = FaceRecognition(img);
                    MCvFont mCvFont = new MCvFont(FONT.CV_FONT_HERSHEY_PLAIN, 20, 20);
                    label1.Text = name;
                    //break;
                }

                imageBox1.Image = mat;

            });
        }

        private string FaceRecognition(Image<Gray, byte> img)
        {
            if (facesAvailable.Count != 0)
            {
                MCvTermCriteria termCrit = new MCvTermCriteria(facesAvailable.Count, 0.001);

                EigenObjectRecognizer recognizer = new EigenObjectRecognizer(facesAvailable.Select(x => x.Value).ToArray(),
                    facesAvailable.Select(x => x.Key).ToArray(),
                    1000, ref termCrit);

                var name = recognizer.Recognize(img.Resize(100, 100, INTER.CV_INTER_CUBIC));
                return Regex.Replace(name.Split('.')[0], "\\(.+?\\)", "");
            }
            return "";
        }
    }
}
