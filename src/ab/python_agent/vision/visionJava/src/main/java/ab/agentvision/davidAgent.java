package ab.agentvision;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;


import ab.planner.TrajectoryPlanner;
import ab.vision.ABObject;
import ab.vision.ABType;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.Vision;

import javax.imageio.ImageIO;
import java.io.FileWriter;
import java.io.IOException;  // Import the IOException class to handle errors
import java.util.List;


public class davidAgent {
    /***************************
     ** ANGRYBIRDS AI AGENT FRAMEWORK
     ** Copyright (c) 2014, XiaoYu (Gary) Ge, Stephen Gould, Jochen Renz
     **  Sahan Abeyasinghe,Jim Keys,  Andrew Wang, Peng Zhang
     ** All rights reserved.
     **This work is licensed under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
     **To view a copy of this license, visit http://www.gnu.org/licenses/
     ***************************/






        private Random randomGenerator;
        public int currentLevel = 1;
        public static int time_limit = 12;
        private Map<Integer,Integer> scores = new LinkedHashMap<Integer,Integer>();
        TrajectoryPlanner tp;
        private boolean firstShot;
        private Point prevTarget;
        // a standalone implementation of the Naive Agent
        public davidAgent() {

            tp = new TrajectoryPlanner();
            prevTarget = null;
            firstShot = true;
            randomGenerator = new Random();
            // --- go to the Poached Eggs episode level selection page ---

        }


        // run the client
        public void run(BufferedImage img, BufferedImage zoomImg) {
            try{
                solve(img,zoomImg);
            } catch (IOException e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                System.exit(-1);
            }
        }

        private double distance(Point p1, Point p2) {
            return Math
                    .sqrt((double) ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y)
                            * (p1.y - p2.y)));
        }


        public void solve(BufferedImage screenshot, BufferedImage zoomScreenshot) throws IOException
        {
            FileWriter myWriter;
            String text;
            myWriter = new FileWriter("filename.txt");
            // process image
            Vision vision = new Vision(screenshot);
            Vision zoonVision = new Vision(zoomScreenshot);

            //find bird on slingshot

            // find the slingshot
            Rectangle sling = vision.findSlingshotMBR();
            if (sling != null) {
                text = String.format("slingshot %d %d %d %d\n", sling.x, sling.y, sling.width, sling.height);
                myWriter.write(text);
            }

            List<ABObject> hills = vision.findHills();
            for (int i = 0; i < hills.size(); i++) {
                ABObject hill = hills.get(i);
                text = String.format("hill %d %d %d %d\n", hill.x , hill.y, hill.width, hill.height);
                myWriter.write(text);
            }
            String obj_type = "";
            List<ABObject> blocks = vision.findBlocksMBR();
            for (int i = 0; i < blocks.size(); i++) {
                ABObject block = blocks.get(i);
                switch (block.getType())
                {
                    case Ice:
                        obj_type = "Ice"; break;
                    case Stone:
                        obj_type = "Stone"; break;
                    case Wood:
                        obj_type = "Wood"; break;
                    default:
                        obj_type = "default";
                }
                text = String.format("%s %d %d %d %d\n", obj_type, block.x , block.y, block.width, block.height);
                myWriter.write(text);
            }

            // get all the pigs
            List<ABObject> pigs = vision.findPigsMBR();

            for (int i = 0; i < pigs.size(); i++) {
                Rectangle pig = pigs.get(i);
                text = String.format("pig %d %d %d %d\n", pig.x , pig.y, pig.width, pig.height);
                myWriter.write(text);
            }

            List<ABObject> tnts = vision.findTNTs();
            for (int i = 0; i < tnts.size(); i++) {
                ABObject tnt = tnts.get(i);
                text = String.format("tnt %d %d %d %d\n", tnt.x , tnt.y, tnt.width, tnt.height);
                myWriter.write(text);
            }

            // Find bird on slingshot
            List<ABObject> _birds = zoonVision.findBirdsMBR();
            if(_birds.isEmpty()){
                System.out.println("An error occurred.");

            }
            else {
                Collections.sort(_birds, new Comparator<Rectangle>() {

                    @Override
                    public int compare(Rectangle o1, Rectangle o2) {

                        return ((Integer) (o1.y)).compareTo((Integer) (o2.y));
                    }
                });

                switch (_birds.get(0).getType()) {
                    case RedBird:
                        obj_type = "RedBird";
                        break;
                    case YellowBird:
                        obj_type = "YellowBird";
                        break;
                    case WhiteBird:
                        obj_type = "WhiteBird";
                        break;
                    case BlackBird:
                        obj_type = "BlackBird";
                        break;
                    case BlueBird:
                        obj_type = "BlueBird";
                        break;
                    default:
                        obj_type = "default";
                }


            }
            text = String.format("%s\n", obj_type);
            myWriter.write(text);
            myWriter.close();
            System.exit(0);
        }

        public static void main(String[] args) throws IOException {
            if (args.length > 0) {
                String imageDirectory = args[0];  // Get the passed image directory
                String zoomImageDirectory = args[1];  // Get the passed image directory
                BufferedImage bufferedImage = ImageIO.read(Files.newInputStream(Paths.get(System.getProperty("user.dir") + imageDirectory)));
                BufferedImage bufferedZoomImage = ImageIO.read(Files.newInputStream(Paths.get(System.getProperty("user.dir") + zoomImageDirectory)));
                davidAgent na = new davidAgent();
                na.run(bufferedImage,bufferedZoomImage);

                // Use the imageDirectory in your application logic
                System.out.println("Received image directory: " + imageDirectory);

                // You can use this directory to access the saved image as needed
            } else {
                System.out.println("No image directory provided.");
            }

        }
}



