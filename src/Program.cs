using System.Drawing;
using WarpTest;
using Emgu.CV;

var options = CmdLine.ProcessArgs(args);
if (!CmdLine.ValidateOptions(options)) return;

// Translated example of https://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection

Mat imgsrc;
using (var img = CvInvoke.Imread(options.ImagePath))
{
    if (options.BorderSize.Width > 0 || options.BorderSize.Height > 0)
    {
        // Add a border around the source image.
        int bw = options.BorderSize.Width;
        int bh = options.BorderSize.Height;
        imgsrc = new Mat(new Size(bw * 2, bh * 2), Emgu.CV.CvEnum.DepthType.Cv8U, 3);
        CvInvoke.CopyMakeBorder(img, imgsrc, bh, bh, bw, bw, Emgu.CV.CvEnum.BorderType.Constant);
    }
    else
    {
        // No border specified.
        imgsrc = img.Clone();
    }
}
// Display the source image in a window.
CvInvoke.Imshow("original", imgsrc);

//rows=height, cols=width eg (y,x), first dim=x, second dim=y for uv map
using var mapX = new Matrix<float>(imgsrc.Height, imgsrc.Width);
using var mapY = new Matrix<float>(imgsrc.Height, imgsrc.Width);

// Are we mapping along the x-axis or y-axis? Set our mapping function.
// Func<PointF, int, int, float, PointF> map = options.MapXAxis ? MapCylinderX : MapCylinderY; // TODO: add command line interface option to change usage of cylinder versus sphere functions
Func<PointF, int, int, float, PointF> map = options.MapXAxis ? MapSphereX : MapSphereY;

// Are we mapping a convex or concave surface?
float zDir = options.Convex ? -1f : 1f;

// Calculate the x & y values.
for (int y = 0; y < imgsrc.Height; y++)
{
    for (int x = 0; x < imgsrc.Width; x++)
    {
        var pointOriginal = new PointF(x, y);
        var pointMapped = map(pointOriginal, imgsrc.Width, imgsrc.Height, zDir);

        mapX[y, x] = pointMapped.X;
        mapY[y, x] = pointMapped.Y;
    }
}

using var imgdest = imgsrc.Clone();

// Remap the source image using our calculated coordinates.
CvInvoke.Remap(imgsrc, imgdest, mapX, mapY, Emgu.CV.CvEnum.Inter.Linear);

// Display the re-mapped image in a window.
CvInvoke.Imshow("remapped", imgdest);

CvInvoke.WaitKey();

static PointF MapCylinderY(PointF point, int w, int h, float zDir)
{
    // Center the point at (0,0)
    var pc = new PointF(point.X - w / 2, point.Y - h / 2);

    // Free parameters that can be adjust for interesting effects.
    float f = w * zDir;
    float r = w;

    float omega = w / 2;
    float z0 = f - (float)Math.Sqrt(f * f - omega * omega);

    float zc = (float)((2.0 * z0 + Math.Sqrt(4.0 * z0 * z0 - 4.0 * (pc.X * pc.X / (f * f) + 1.0) * (z0 * z0 - r * r))) / (2.0 * (pc.X * pc.X / (f * f) + 1.0)));

    // The calculation below skews the image.
    var final_point = new PointF(pc.X * zc / f, pc.Y * zc / f);

    final_point.X += w / 2;
    final_point.Y += h / 2;

    return final_point;
}

// Copied the MapCylinderY(..) function and changed a few lines rather than
// having one function with a bunch of if (mapX) ... else ...tests inside the loop.
static PointF MapCylinderX(PointF point, int w, int h, float zDir)
{
    // Center the point at (0,0)
    var pc = new PointF(point.X - w / 2, point.Y - h / 2);

    // Free parameters that can be adjust for interesting effects.
    float f = h * zDir;
    float r = h;

    float omega = h / 2;
    float z0 = f - (float)Math.Sqrt(f * f - omega * omega);

    float zc = (float)((2.0 * z0 + Math.Sqrt(4.0 * z0 * z0 - 4.0 * (pc.Y * pc.Y / (f * f) + 1.0) * (z0 * z0 - r * r))) / (2.0 * (pc.Y * pc.Y / (f * f) + 1.0)));

    // The calculation below skews the image.
    var final_point = new PointF(pc.X * zc / f, pc.Y * zc / f);

    final_point.X += w / 2;
    final_point.Y += h / 2;

    return final_point;
}

static PointF MapSphereY(PointF point, int w, int h, float zDir)
{
    // Center the point at (0,0)
    var pc = new PointF(point.X - w / 2, point.Y - h / 2);

    // set the free parameters
    float f = w * zDir; // NOTE: this is setting the distance from the camera to the image reflection plane to the width of the image
    float r = w; // NOTE: this is setting the radius of the cylinder to the width of the image

    float z0 = f - (float)Math.Sqrt(r * r - (w / 2) * (w / 2) - (h / 2) * (h / 2)); // NOTE: this is the location of the center of the sphere along the z-axis
    float a = (float)(1.0 + pc.X * pc.X / (f * f) + pc.Y * pc.Y / (f * f)); // NOTE: this is the coefficient on the z^2 term in the quadratic equation solution for z as function of ximg and yimg

    float zc = (float)((2.0 * z0 + Math.Sqrt(4.0 * z0 * z0 - 4.0 * a * (z0 * z0 - r * r))) / (2.0 * a));

    // The calculation below skews the image.
    var final_point = new PointF(pc.X * zc / f, pc.Y * zc / f);

    final_point.X += w / 2;
    final_point.Y += h / 2;

    return final_point;
}

static PointF MapSphereX(PointF point, int w, int h, float zDir)
{
    // Center the point at (0,0)
    var pc = new PointF(point.X - w / 2, point.Y - h / 2);

    // set the free parameters
    float f = h * zDir; // NOTE: this is setting the distance from the camera to the image reflection plane to the height of the image
    float r = h; // NOTE: this is setting the radius of the cylinder to the height of the image

    float z0 = f - (float)Math.Sqrt(r * r - (w / 2) * (w / 2) - (h / 2) * (h / 2)); // NOTE: this is the location of the center of the sphere along the z-axis
    float a = (float)(1.0 + pc.X * pc.X / (f * f) + pc.Y * pc.Y / (f * f)); // NOTE: this is the coefficient on the z^2 term in the quadratic equation solution for z as function of ximg and yimg

    float zc = (float)((2.0 * z0 + Math.Sqrt(4.0 * z0 * z0 - 4.0 * a * (z0 * z0 - r * r))) / (2.0 * a));

    // The calculation below skews the image.
    var final_point = new PointF(pc.X * zc / f, pc.Y * zc / f);

    final_point.X += w / 2;
    final_point.Y += h / 2;

    return final_point;
}
