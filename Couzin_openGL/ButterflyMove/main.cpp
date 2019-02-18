#include <string>
#include <GL/glui.h>
#include <ctime>
#include "Model.h"

float xy_aspect = 1.0;
float zoom = 0;

/** These are the live variables passed into GLUI ***/
int autoTime = 0;
int autoWeight = 0;
float deltaT = 0.01;
float borderSize = 1.0;
bool pause = false;
float view_rotate[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
int directionColor = 1;

clock_t last_time = 0;

int   main_window;
float scale = 1.0;


/** Pointers to the windows and some of the controls we'll create **/
GLUI *glui, *glui2;

/********** User IDs for callbacks ********/
#define SECOND_ID            400
#define AUTOTIME_ID          401
#define PAUSE_ID			 402
#define RESET_ID			 403
#define VIEW_RESET_ID		 404
#define AUTOWEIGHT_ID		 405
#define COLOR_ID			 406
#define POSITION_RESET_ID	 407


/********** Miscellaneous global variables **********/

GLfloat light0_ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
GLfloat light0_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat light0_position[] = { 0.0f, 2.0f, 3.0f, 0.0f };

/**************************************** control_cb() *******************/
/* GLUI control callback                                                 */

void control_cb(int control)
{
	if (control == PAUSE_ID) {
		pause = !pause;
	}
	else if (control == RESET_ID) {
		reset();
	}
	else if (control == VIEW_RESET_ID) {
		view_rotate[0] = 1; view_rotate[4] = 0; view_rotate[8] = 0; view_rotate[12] = 0;
		view_rotate[1] = 0; view_rotate[5] = 1; view_rotate[9] = 0; view_rotate[13] = 0;
		view_rotate[2] = 0; view_rotate[6] = 0; view_rotate[10] = 1; view_rotate[14] = 0;
		view_rotate[3] = 0; view_rotate[7] = 0; view_rotate[11] = 0; view_rotate[15] = 1;
	}
	else if (control == POSITION_RESET_ID) {
		positionReset();
	}
}

/**************************************** myGlutKeyboard() **********/

void myGlutKeyboard(unsigned char Key, int x, int y)
{
	switch (Key)
	{
	case 27:
	case 'q':
		exit(0);
		break;
	case 'z':
		zoom += 0.2;
		break;
	case 'x':
		zoom -= 0.2;
		break;
	};

	glutPostRedisplay();
}


/***************************************** myGlutMenu() ***********/

void myGlutMenu(int value)
{
	myGlutKeyboard(value, 0, 0);
}


/***************************************** myGlutIdle() ***********/

void myGlutIdle(void)
{
	/* According to the GLUT specification, the current window is
	undefined during an idle callback.  So we need to explicitly change
	it if necessary */
	if (glutGetWindow() != main_window)
		glutSetWindow(main_window);

	/*  GLUI_Master.sync_live_all();  -- not needed - nothing to sync in this
	application  */

	glutPostRedisplay();
}

/***************************************** myGlutMouse() **********/

void myGlutMouse(int button, int button_state, int x, int y)
{
}


/***************************************** myGlutMotion() **********/

void myGlutMotion(int x, int y)
{
	glutPostRedisplay();
}

/**************************************** myGlutReshape() *************/

void myGlutReshape(int x, int y)
{
	int tx, ty, tw, th;
	GLUI_Master.get_viewport_area(&tx, &ty, &tw, &th);
	glViewport(tx, ty, tw, th);

	xy_aspect = (float)tw / (float)th;

	glutPostRedisplay();
}

void draw_axes(float scale)
{
	//glDisable(GL_LIGHTING);

	glPushMatrix();
	glScalef(scale, scale, scale);
	glTranslatef(-1, -1, -1);

	glLineWidth(3.0f);
	glBegin(GL_LINES);
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(2.3f, 0.05f, 0.0);  glVertex3f(2.5, 0.25f, 0.0); /* Letter X */
	glVertex3f(2.3f, .25f, 0.0);  glVertex3f(2.5, 0.05f, 0.0);
	glEnd();

	glPushMatrix();	/* Letter Y */
	float len = 0.1;
	glTranslatef(0.2, 2.5, 0);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, -1.5 * len, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(-len, len, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(len, len, 0.0f);
	glEnd();
	glPopMatrix();

	glPushMatrix();	/* Letter Z */
	len = 0.1;
	glTranslatef(-0.2, 0, 2);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(len, 0.0f, 0.0f);
	glVertex3f(len, 0.0f, 0.0f);
	glVertex3f(0.0f, -len, 0.0f);
	glVertex3f(0.0f, -len, 0.0f);
	glVertex3f(len, -len, 0.0f);
	glEnd();
	glPopMatrix();

	glLineWidth(1.0f);
	glBegin(GL_LINES);
	glColor3f(0.4, 0.0, 0.0);
	glVertex3f(0.0, 0.0, 0.0);  glVertex3f(10.0, 0.0, 0.0); /* X axis      */

	glColor3f(0.0, 0.4, 0.0);
	glVertex3f(0.0, 0.0, 0.0);  glVertex3f(0.0, 10.0, 0.0); /* Y axis      */

	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(0.0, 0.0, 0.0);  glVertex3f(0.0, 0.0, 10.0); /* Z axis    */
	glEnd();

	glPopMatrix();

	//glEnable(GL_LIGHTING);
}

/***************************************** myGlutDisplay() *****************/

void myGlutDisplay(void)
{
	// clear screen
	glClearColor(.9f, .9f, .9f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view settings:
	glLoadIdentity();

	// camera setting
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(50, xy_aspect, 1.0, 10.0);
	glMatrixMode(GL_MODELVIEW);
	gluLookAt(0.0, 0.0, 3.5 + zoom,  /* eye is at (0,0,5) */
		0.0, 0.0, 0.0,      /* center is at (0,0,0) */
		0.0, 1.0, 0.);      /* up is in positive Y direction */

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);

	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);

	draw_axes(1.0f);

	glPushMatrix();
	glMultMatrixf(view_rotate);
	for (int sp = 0; sp < SPECIES_SIZE; sp++) {
		if (speciesEnable[sp] == 0) continue;

		for (int i = 0; i < number[sp]; i++) {
			glPushMatrix();
			float color0, color1, color2;
			if (directionColor) {
				Utils::shade(color0, color1, color2, (directions[sp][i * 3] / 2.0 + 0.5), (directions[sp][i * 3 + 1] / 2.0 + 0.5), (directions[sp][i * 3 + 2] / 2.0 + 0.5), sp);
			}
			else {
				Utils::shade(color0, color1, color2, (positions[sp][i * 3] / borderSize / 2.0 + 0.5), (positions[sp][i * 3 + 1] / borderSize / 2.0 + 0.5), (positions[sp][i * 3 + 2] / borderSize / 2.0 + 0.5), sp);
			}

			float r = 0.01f;
			//if (sp == TWO) r *= 5;
			glColor3f(color0, color1, color2);
			glTranslatef(positions[sp][i * 3], positions[sp][i * 3 + 1], positions[sp][i * 3 + 2]);
			glutSolidSphere(r, 32, 16);
			glPopMatrix();
		}
	}
	glPopMatrix();

	glEnable(GL_LIGHTING);

	double dt = deltaT;
	clock_t timeT = clock();
	if (autoTime && last_time != 0) {
		dt = (timeT - last_time) / 1000.0f;
		last_time = timeT;
	}
	else if (autoTime) {
		dt = 0;
	}
	last_time = timeT;

	if (!pause)
		update(dt);

	glutSwapBuffers();
}


/**************************************** main() ********************/

int main(int argc, char* argv[])
{
	/*** initialize the model ***/
	initialize();

	/****************************************/
	/*   Initialize GLUT and create window  */
	/****************************************/

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(50, 50);
	glutInitWindowSize(1200, 800);
	xy_aspect = (double)1200 / 800;

	main_window = glutCreateWindow("GLUI Example 5");
	glutDisplayFunc(myGlutDisplay);
	GLUI_Master.set_glutReshapeFunc(myGlutReshape);
	GLUI_Master.set_glutKeyboardFunc(myGlutKeyboard);
	GLUI_Master.set_glutSpecialFunc(NULL);
	GLUI_Master.set_glutMouseFunc(myGlutMouse);
	glutMotionFunc(myGlutMotion);

	glClearColor(.5f, .5f, .5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/****************************************/
	/*       Set up OpenGL lights           */
	/****************************************/

	glEnable(GL_LIGHTING);
	glEnable(GL_NORMALIZE);

	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);

	/****************************************/
	/*          Enable z-buferring          */
	/****************************************/

	glEnable(GL_DEPTH_TEST);

	/****************************************/
	/*         Here's the GLUI code         */
	/****************************************/

	printf("GLUI version: %3.2f\n", GLUI_Master.get_version());

	/*** Create the side subwindow ***/
	glui = GLUI_Master.create_glui_subwindow(main_window,
		GLUI_SUBWINDOW_RIGHT);

	/***** Control for One species params *****/
	GLUI_Panel *first_species = new GLUI_Rollout(glui, "Species", true);

	new GLUI_Checkbox(first_species, "Enable first spicies", &speciesEnable[ONE], 1, control_cb);
	GLUI_Spinner *firstSpinnerSpeed =
		new GLUI_Spinner(first_species, "speed:", &speed[ONE]);
	firstSpinnerSpeed->set_float_limits(0.5, 2);
	firstSpinnerSpeed->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *firstSpinnerCount =
		new GLUI_Spinner(first_species, "Number:", &number[ONE]);
	firstSpinnerCount->set_int_limits(0, 2000);
	firstSpinnerCount->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *firstSpinnerZOR =
		new GLUI_Spinner(first_species, "ZOR:", &zor[ONE]);
	firstSpinnerZOR->set_float_limits(0.0f, 10.0);
	firstSpinnerZOR->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *firstSpinnerZOO =
		new GLUI_Spinner(first_species, "ZOO:", &zoo[ONE]);
	firstSpinnerZOO->set_float_limits(0.0f, 10.0);
	firstSpinnerZOO->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *firstSpinnerZOA =
		new GLUI_Spinner(first_species, "ZOA:", &zoa[ONE]);
	firstSpinnerZOA->set_float_limits(0.0f, 10.0);
	firstSpinnerZOA->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *firstSpinnerBlind =
		new GLUI_Spinner(first_species, "Blind Angle:", &blind[ONE]);
	firstSpinnerBlind->set_float_limits(0.0f, 360.0f);
	firstSpinnerBlind->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *firstSpinnerTurn =
		new GLUI_Spinner(first_species, "Turn Angle:", &turn[ONE]);
	firstSpinnerTurn->set_float_limits(0.1f, 360.0f);
	firstSpinnerTurn->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *firstSpinnerWeight =
		new GLUI_Spinner(first_species, "ZOO - ZOA weight:", &zooaWeights[ONE]);
	firstSpinnerWeight->set_float_limits(0.0f, 1.0f);
	firstSpinnerWeight->set_alignment(GLUI_ALIGN_RIGHT);


	/***** Control for Second species params *****/
	new GLUI_Column(first_species, false);
	new GLUI_Checkbox(first_species, "Enable second spicies", &speciesEnable[TWO], SECOND_ID, control_cb);
	GLUI_Spinner *secondSpinnerSpeed =
		new GLUI_Spinner(first_species, "speed:", &speed[TWO]);
	secondSpinnerSpeed->set_float_limits(0.5, 2);
	secondSpinnerSpeed->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *secondSpinnerCount =
		new GLUI_Spinner(first_species, "Number:", &number[TWO]);
	secondSpinnerCount->set_int_limits(0, 2000);
	secondSpinnerCount->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *secondSpinnerZOR =
		new GLUI_Spinner(first_species, "ZOR:", &zor[TWO]);
	secondSpinnerZOR->set_float_limits(0.0f, 4.0);
	secondSpinnerZOR->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *secondSpinnerZOO =
		new GLUI_Spinner(first_species, "ZOO:", &zoo[TWO]);
	secondSpinnerZOO->set_float_limits(0.0f, 4.0);
	secondSpinnerZOO->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *secondSpinnerZOA =
		new GLUI_Spinner(first_species, "ZOA:", &zoa[TWO]);
	secondSpinnerZOA->set_float_limits(0.0f, 4.0);
	secondSpinnerZOA->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *secondSpinnerBlind =
		new GLUI_Spinner(first_species, "Blind Angle:", &blind[TWO]);
	secondSpinnerBlind->set_float_limits(0.0f, 360.0f);
	secondSpinnerBlind->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *secondSpinnerTurn =
		new GLUI_Spinner(first_species, "Turn Angle:", &turn[TWO]);
	secondSpinnerTurn->set_float_limits(0.1f, 360.0f);
	secondSpinnerTurn->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *secondSpinnerWeight =
		new GLUI_Spinner(first_species, "ZOO - ZOA weight:", &zooaWeights[TWO]);
	secondSpinnerWeight->set_float_limits(0.0f, 1.0f);
	secondSpinnerWeight->set_alignment(GLUI_ALIGN_RIGHT);


	/*** Add another rollout for Global Settings***/
	GLUI_Panel *species = new GLUI_Rollout(glui, "Species Settings", true);

	GLUI_Spinner *oneTwoSpinnerZOR =
		new GLUI_Spinner(species, "1 - 2 ZOR:", &intraZOR[ONE]);
	oneTwoSpinnerZOR->set_float_limits(0.0f, 4.0);
	oneTwoSpinnerZOR->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *oneTwoSpinnerZOO =
		new GLUI_Spinner(species, "1 - 2 ZOO:", &intraZOO[ONE]);
	oneTwoSpinnerZOO->set_float_limits(0.0f, 4.0);
	oneTwoSpinnerZOO->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *oneTwoSpinnerZOA =
		new GLUI_Spinner(species, "1 - 2 ZOA:", &intraZOA[ONE]);
	oneTwoSpinnerZOA->set_float_limits(0.0f, 4.0);
	oneTwoSpinnerZOA->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *oneTwoSpinnerZOAWeight =
		new GLUI_Spinner(species, "1 - 2 ZOO - ZOA weight:", &zooaWeightsIntra[ONE]);
	oneTwoSpinnerZOAWeight->set_float_limits(0.0f, 1.0f);
	oneTwoSpinnerZOAWeight->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *oneTwoSpinnerWeight =
		new GLUI_Spinner(species, "1 - 2 total weight:", &intraWeights[ONE]);
	oneTwoSpinnerWeight->set_float_limits(0.0f, 1.0f);
	oneTwoSpinnerWeight->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *twoOneSpinnerZOR =
		new GLUI_Spinner(species, "2 - 1 ZOR:", &intraZOR[TWO]);
	twoOneSpinnerZOR->set_float_limits(0.0f, 4.0);
	twoOneSpinnerZOR->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *twoOneSpinnerZOO =
		new GLUI_Spinner(species, "2 - 1 ZOO:", &intraZOO[TWO]);
	twoOneSpinnerZOO->set_float_limits(0.0f, 4.0);
	twoOneSpinnerZOO->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *twoOneSpinnerZOA =
		new GLUI_Spinner(species, "2 - 1 ZOA:", &intraZOA[TWO]);
	twoOneSpinnerZOA->set_float_limits(0.0f, 4.0);
	twoOneSpinnerZOA->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *twoOneSpinnerZOAWeight =
		new GLUI_Spinner(species, "2 - 1 ZOO - ZOA weight:", &zooaWeightsIntra[TWO]);
	twoOneSpinnerZOAWeight->set_float_limits(0.0f, 1.0f);
	twoOneSpinnerZOAWeight->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *twoOneSpinnerWeight =
		new GLUI_Spinner(species, "2 - 1 total weight:", &intraWeights[TWO]);
	twoOneSpinnerWeight->set_float_limits(0.0f, 1.0f);
	twoOneSpinnerWeight->set_alignment(GLUI_ALIGN_RIGHT);

	new GLUI_Column(species, false);
	new GLUI_Checkbox(species, "Auto delta T", &autoTime, AUTOTIME_ID, control_cb);

	GLUI_Spinner *timeSpinner =
		new GLUI_Spinner(species, "deltaT:", &deltaT);
	timeSpinner->set_float_limits(0.001f, 0.5f);
	timeSpinner->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Spinner *borderSpinner =
		new GLUI_Spinner(species, "Border Size:", &borderSize);
	borderSpinner->set_float_limits(1.0f, 4.0);
	borderSpinner->set_alignment(GLUI_ALIGN_RIGHT);

	new GLUI_Checkbox(species, "Auto weighted", &autoWeight, AUTOWEIGHT_ID, control_cb);

	new GLUI_Checkbox(species, "Direction Shade", &directionColor, COLOR_ID, control_cb);

	/*** Disable/Enable buttons ***/

	new GLUI_Button(glui, "Pause/Start", PAUSE_ID, control_cb);

	new GLUI_Button(glui, "Reset", RESET_ID, control_cb);

	GLUI_Rotation *view_rot = new GLUI_Rotation(glui, "Objects", view_rotate);
	view_rot->set_spin(1.0);

	new GLUI_Button(glui, "View Reset", VIEW_RESET_ID, control_cb);
	new GLUI_Button(glui, "Position Reset", POSITION_RESET_ID, control_cb);

	/****** A 'quit' button *****/
	new GLUI_Button(glui, "Quit", 0, (GLUI_Update_CB)exit);


	/*** Create the bottom subwindow ***/
//	glui2 = GLUI_Master.create_glui_subwindow(main_window,
//		GLUI_SUBWINDOW_BOTTOM);
//	glui2->set_main_gfx_window(main_window);

	///*** Add another rollout for Global Settings***/
	//GLUI_Panel *timeSettings = new GLUI_Panel(glui2, "Time Setting:", true);

	//new GLUI_Checkbox(timeSettings, "Auto delta T", &autoTime, AUTOTIME_ID, control_cb);

	//GLUI_Spinner *timeSpinner =
	//	new GLUI_Spinner(timeSettings, "deltaT:", &deltaT);
	//timeSpinner->set_float_limits(0.001f, 0.5f);
	//timeSpinner->set_alignment(GLUI_ALIGN_RIGHT);

	//GLUI_Spinner *borderSpinner =
	//	new GLUI_Spinner(timeSettings, "Border Size:", &borderSize);
	//borderSpinner->set_float_limits(1.0f, 4.0);
	//borderSpinner->set_alignment(GLUI_ALIGN_RIGHT);

	//new GLUI_Checkbox(timeSettings, "Auto weighted", &autoWeight, AUTOWEIGHT_ID, control_cb);

	//new GLUI_Checkbox(timeSettings, "Direction Shade", &directionColor, COLOR_ID, control_cb);


	/**** Link windows to GLUI, and register idle callback ******/

	glui->set_main_gfx_window(main_window);

	/**** We register the idle callback with GLUI, *not* with GLUT ****/
	GLUI_Master.set_glutIdleFunc(myGlutIdle);

	/**** Regular GLUT main loop ****/

	glutMainLoop();

	return EXIT_SUCCESS;
}

